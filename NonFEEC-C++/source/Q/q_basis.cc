#include <Q/q_basis.h>

namespace Q
{
  using namespace dealii;

  QBasis::QBasis(const ParametersMs &parameters_ms,
                 const std::string & parameter_filename,
                 typename Triangulation<3>::active_cell_iterator &global_cell,
                 CellId                                           first_cell,
                 unsigned int local_subdomain,
                 MPI_Comm     mpi_communicator)
    : mpi_communicator(mpi_communicator)
    , parameters(parameters_ms)
    , parameter_filename(parameter_filename)
    , fe(1)
    , dof_handler(triangulation)
    , constraints_vector(GeometryInfo<3>::vertices_per_cell)
    , corner_points(GeometryInfo<3>::vertices_per_cell)
    , filename_global("")
    , solution_vector(GeometryInfo<3>::vertices_per_cell)
    , global_element_matrix(fe.dofs_per_cell, fe.dofs_per_cell)
    , is_built_global_element_matrix(false)
    , global_element_rhs(fe.dofs_per_cell)
    , global_weights(fe.dofs_per_cell, 0)
    , is_set_global_weights(false)
    , global_cell_id(global_cell->id())
    , first_cell(first_cell)
    , local_subdomain(local_subdomain)
    , volume_measure(0)
    , face_measure(GeometryInfo<3>::faces_per_cell, 0)
    , edge_measure(GeometryInfo<3>::lines_per_cell, 0)
    , basis_q1(global_cell)
  {
    // set corner points
    for (unsigned int vertex_n = 0;
         vertex_n < GeometryInfo<3>::vertices_per_cell;
         ++vertex_n)
      {
        corner_points[vertex_n] = global_cell->vertex(vertex_n);
      }

    volume_measure = global_cell->measure();

    for (unsigned int j_face = 0; j_face < GeometryInfo<3>::faces_per_cell;
         ++j_face)
      {
        face_measure[j_face] = global_cell->face(j_face)->measure();
      }

    for (unsigned int j_egde = 0; j_egde < GeometryInfo<3>::lines_per_cell;
         ++j_egde)
      {
        edge_measure[j_egde] = global_cell->line(j_egde)->measure();
      }
  }



  QBasis::QBasis(const QBasis &X)
    : mpi_communicator(X.mpi_communicator)
    , parameters(X.parameters)
    , parameter_filename(X.parameter_filename)
    , triangulation()
    , fe(X.fe)
    , dof_handler(triangulation)
    , constraints_vector(X.constraints_vector)
    , corner_points(X.corner_points)
    //    , sparsity_pattern(X.sparsity_pattern) // only possible if object is
    //    empty , diffusion_matrix(X.diffusion_matrix) // only possible if
    //    object is empty , system_matrix(X.system_matrix)       // only
    //    possible if object is empty
    , filename_global(X.filename_global)
    , solution_vector(X.solution_vector)
    , global_rhs(X.global_rhs)
    , system_rhs(X.system_rhs)
    , global_element_matrix(X.global_element_matrix)
    , is_built_global_element_matrix(X.is_built_global_element_matrix)
    , global_element_rhs(X.global_element_rhs)
    , global_weights(X.global_weights)
    , is_set_global_weights(X.is_set_global_weights)
    , global_solution(X.global_solution)
    , global_cell_id(X.global_cell_id)
    , first_cell(X.first_cell)
    , local_subdomain(X.local_subdomain)
    , volume_measure(X.volume_measure)
    , face_measure(X.face_measure)
    , edge_measure(X.edge_measure)
    , basis_q1(X.basis_q1)
  {}



  void
    QBasis::make_grid()
  {
    GridGenerator::general_cell(triangulation,
                                corner_points,
                                /* colorize faces */ false);

    triangulation.refine_global(parameters.n_refine_local);
  }


  QBasis::~QBasis()
  {
    system_matrix.clear();
    for (unsigned int n_basis = 0; n_basis < solution_vector.size(); ++n_basis)
      {
        constraints_vector[n_basis].clear();
      }
    dof_handler.clear();
  }


  void
    QBasis::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    if (parameters.verbose)
      std::cout << "Global cell id  " << global_cell_id.to_string()
                << " (subdomain = " << local_subdomain
                << "):   " << triangulation.n_active_cells()
                << " active fine cells --- " << dof_handler.n_dofs()
                << " subgrid dof" << std::endl;

    /*
     * Set up Dirichlet boundary conditions and sparsity pattern.
     */
    DynamicSparsityPattern dsp(dof_handler.n_dofs());

    for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<3>::vertices_per_cell;
         ++index_basis)
      {
        basis_q1.set_index(index_basis);

        constraints_vector[index_basis].clear();
        DoFTools::make_hanging_node_constraints(
          dof_handler, constraints_vector[index_basis]);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          /*boundary id*/ 0,
          basis_q1,
          constraints_vector[index_basis]);
        constraints_vector[index_basis].close();
      }

    DoFTools::make_sparsity_pattern(
      dof_handler,
      dsp,
      constraints_vector[0], // sparsity pattern is the same for each basis
      /*keep_constrained_dofs =*/true); // for time stepping this is essential
                                        // to be true
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    diffusion_matrix.reinit(sparsity_pattern);

    for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<3>::vertices_per_cell;
         ++index_basis)
      {
        solution_vector[index_basis].reinit(dof_handler.n_dofs());
      }
    system_rhs.reinit(dof_handler.n_dofs());
    global_rhs.reinit(dof_handler.n_dofs());
  }



  void
    QBasis::assemble_system()
  {
    QGauss<3> quadrature_formula(fe.degree + 1);

    FEValues<3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_diffusion_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    /*
     * Matrix coefficient and vector to store the values.
     */
    const EquationData::Diffusion_A matrix_coeff(parameter_filename);
    std::vector<Tensor<2, 3>>       matrix_coeff_values(n_q_points);

    /*
     * Right hand side and vector to store the values.
     */
    const EquationData::RightHandSideParsed right_hand_side(
      parameter_filename, /* n_components */ 1);
    std::vector<double> rhs_values(n_q_points);

    /*
     * Integration over cells.
     */
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_diffusion_matrix = 0;
        cell_rhs              = 0;

        fe_values.reinit(cell);

        // Now actually fill with values.
        matrix_coeff.value_list(fe_values.get_quadrature_points(),
                                matrix_coeff_values);
        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    cell_diffusion_matrix(i, j) +=
                      fe_values.shape_grad(i, q_index) *
                      matrix_coeff_values[q_index] *
                      fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index);
                  } // end ++j

                cell_rhs(i) += fe_values.shape_value(i, q_index) *
                               rhs_values[q_index] * fe_values.JxW(q_index);
              } // end ++i
          }     // end ++q_index

        // get global indices
        cell->get_dof_indices(local_dof_indices);
        /*
         * Now add the cell matrix and rhs to the right spots
         * in the global matrix and global rhs. Constraints will
         * be taken care of later.
         */
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                diffusion_matrix.add(local_dof_indices[i],
                                     local_dof_indices[j],
                                     cell_diffusion_matrix(i, j));
              }
            global_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      } // end ++cell
  }



  void
    QBasis::assemble_global_element_matrix()
  {
    // First, reset.
    global_element_matrix = 0;

    // Get lengths of tmp vectors for assembly
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double> tmp(dof_handler.n_dofs());

    // This assembles the local contribution to the global global matrix
    // with an algebraic trick. It uses the local system matrix stored in
    // the respective basis object.
    for (unsigned int i_test = 0; i_test < dofs_per_cell; ++i_test)
      {
        // set an alias name
        const Vector<double> &test_vec = solution_vector[i_test];

        for (unsigned int i_trial = 0; i_trial < dofs_per_cell; ++i_trial)
          {
            // set an alias name
            const Vector<double> &trial_vec = solution_vector[i_trial];

            // tmp = system_matrix*trial_vec
            diffusion_matrix.vmult(tmp, trial_vec);

            // global_element_diffusion_matrix = test_vec*tmp
            global_element_matrix(i_test, i_trial) += (test_vec * tmp);

            // reset
            tmp = 0;
          } // end for i_trial

        global_element_rhs(i_test) += test_vec * global_rhs;

      } // end for i_test

    is_built_global_element_matrix = true;
  }


  void
    QBasis::solve_direct(unsigned int n_basis)
  {
    Timer timer;
    if (parameters.verbose)
      {
        std::cout << "	Solving linear system (SparseDirectUMFPACK) in cell   "
                  << global_cell_id.to_string() << "   for basis   " << n_basis
                  << ".....";

        timer.restart();
      }

    // use direct solver
    SparseDirectUMFPACK A_inv;
    A_inv.initialize(system_matrix);

    A_inv.vmult(solution_vector[n_basis], system_rhs);

    constraints_vector[n_basis].distribute(solution_vector[n_basis]);

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }



  void
    QBasis::solve_iterative(unsigned int index_basis)
  {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<>    solver(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.6);

    solver.solve(system_matrix,
                 solution_vector[index_basis],
                 system_rhs,
                 preconditioner);

    constraints_vector[index_basis].distribute(solution_vector[index_basis]);

    if (parameters.verbose)
      std::cout << "   "
                << "(cell   " << global_cell_id.to_string() << ") "
                << "(basis   " << index_basis << ")   "
                << solver_control.last_step()
                << " fine CG iterations needed to obtain convergence."
                << std::endl;
  }



  const FullMatrix<double> &
    QBasis::get_global_element_matrix() const
  {
    return global_element_matrix;
  }



  const Vector<double> &
    QBasis::get_global_element_rhs() const
  {
    return global_element_rhs;
  }



  const std::string &
    QBasis::get_filename_global()
  {
    return parameters.filename_global;
  }



  void
    QBasis::set_output_flag()
  {
    parameters.set_output_flag(global_cell_id, first_cell);
  }



  void
    QBasis::set_global_weights(const std::vector<double> &weights)
  {
    // Copy assignment of global weights
    global_weights = weights;

    // reinitialize the global solution on this cell
    global_solution.reinit(dof_handler.n_dofs());

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // Set global solution using the weights and the local basis.
    for (unsigned int index_basis = 0; index_basis < dofs_per_cell;
         ++index_basis)
      {
        // global_solution = 1*global_solution +
        // global_weights[index_basis]*solution_vector[index_basis]
        global_solution.sadd(1,
                             global_weights[index_basis],
                             solution_vector[index_basis]);
      }

    is_set_global_weights = true;
  }



  void
    QBasis::set_filename_global()
  {
    parameters.filename_global +=
      ("." + Utilities::int_to_string(local_subdomain, 5) + ".cell-" +
       global_cell_id.to_string() + ".vtu");
  }



  void
    QBasis::output_basis()
  {
    Timer timer;
    if (parameters.verbose)
      {
        std::cout << "	Writing local basis in cell   "
                  << global_cell_id.to_string() << ".....";

        timer.restart();
      }

    for (unsigned int n_basis = 0; n_basis < GeometryInfo<3>::vertices_per_cell;
         ++n_basis)
      {
        Vector<double> &basis = solution_vector[n_basis];

        std::vector<std::string> solution_names(1, "u");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          interpretation(1, DataComponentInterpretation::component_is_scalar);

        DataOut<3> data_out;
        data_out.attach_dof_handler(dof_handler);

        data_out.add_data_vector(basis,
                                 solution_names,
                                 DataOut<3>::type_dof_data,
                                 interpretation);

        data_out.build_patches();

        // filename
        std::string filename = "basis_q";
        filename += "." + Utilities::int_to_string(local_subdomain, 5);
        filename += ".cell-" + global_cell_id.to_string();
        filename += ".index-";
        filename += Utilities::int_to_string(n_basis, 2);
        filename += ".vtu";

        std::ofstream output(parameters.dirname_output + "/" + filename);
        data_out.write_vtu(output);
      }

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }


  void
    QBasis::output_global_solution_in_cell() const
  {
    Assert(is_set_global_weights,
           ExcMessage("Global weights must be set first."));

    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names(1, "u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        1, DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(global_solution,
                             solution_names,
                             DataOut<3>::type_dof_data,
                             data_component_interpretation);

    // Postprocess
    std::unique_ptr<Q_PostProcessor> postprocessor(
      new Q_PostProcessor(parameter_filename));
    data_out.add_data_vector(global_solution, *postprocessor);

    data_out.build_patches();

    std::ofstream output(parameters.dirname_output + "/" +
                         parameters.filename_global);
    data_out.write_vtu(output);
  }



  void
    QBasis::run()
  {
    Timer timer;

    if (true)
      {
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int  name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        std::string proc_name(processor_name, name_len);

        std::cout << "	Solving for basis in cell   "
                  << global_cell_id.to_string() << "   [machine: " << proc_name
                  << " | rank: "
                  << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                  << "]   .....";
        timer.restart();
      }

    make_grid();

    setup_system();

    assemble_system();

    set_filename_global();

    for (unsigned int index_basis = 0;
         index_basis < GeometryInfo<3>::vertices_per_cell;
         ++index_basis)
      {
        // reset everything
        system_rhs.reinit(solution_vector[index_basis].size());
        system_matrix.reinit(sparsity_pattern);

        system_matrix.copy_from(diffusion_matrix);

        // Now take care of constraints
        constraints_vector[index_basis].condense(system_matrix, system_rhs);

        // Now solve
        if (parameters.use_direct_solver)
          solve_direct(index_basis);
        else
          {
            solve_iterative(index_basis);
          }
      }

    assemble_global_element_matrix();

    // We need to set a filename for the global solution on the current cell
    set_filename_global();

    // Write basis output only if desired
    set_output_flag();
    if (parameters.output_flag)
      {
        try
          {
            Tools::create_data_directory(parameters.dirname_output);
          }
        catch (std::runtime_error &e)
          {
            // No exception handling here.
          }
        output_basis();
      }

    {
      // Free memory as much as possible
      system_matrix.clear();
      for (unsigned int i = 0; i < GeometryInfo<3>::vertices_per_cell; ++i)
        {
          constraints_vector[i].clear();
        }
    }

    if (true)
      {
        timer.stop();

        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

} // end namespace Q
