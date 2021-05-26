#include <Q/q_ref.h>

namespace Q
{
  using namespace dealii;


  QStd::QStd(ParametersStd &parameters_, const std::string &parameter_filename_)
    : mpi_communicator(MPI_COMM_WORLD)
    , parameters(parameters_)
    , parameter_filename(parameter_filename_)
    , triangulation(mpi_communicator,
                    typename Triangulation<3>::MeshSmoothing(
                      Triangulation<3>::smoothing_on_refinement |
                      Triangulation<3>::smoothing_on_coarsening))
    , fe(1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}


  QStd::~QStd()
  {
    system_matrix.clear();
    constraints.clear();
    dof_handler.clear();
  }


  void
    QStd::make_grid()
  {
    TimerOutput::Scope t(computing_timer, "mesh generation");

    GridGenerator::hyper_cube(triangulation, 0, 1, /* colorize */ true);

    triangulation.refine_global(parameters.n_refine);
  }



  void
    QStd::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "system setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    if (!parameters.is_pure_neumann)
      {
        // Set up Dirichlet boundary conditions.
        const EquationData::BoundaryValues_u boundary_u;
        for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
          {
            VectorTools::interpolate_boundary_values(
              dof_handler,
              /*boundary id*/ i, // only even boundary id
              boundary_u,
              constraints);
          }
      }

    /*
     * If we have a Laplace problem (not Helmholtz) and a pure
     * Neumann problem then we need to make sure that u is unique.
     * We therefore add a constraint on dofs
     */
    if (parameters.is_laplace && parameters.is_pure_neumann)
      {
        IndexSet locally_relevant_boundary_dofs,
          first_boundary_index_on_rank_zero;

        DoFTools::extract_boundary_dofs(dof_handler,
                                        ComponentMask(),
                                        locally_relevant_boundary_dofs);

        // initially set a non-admissible value
        unsigned int first_local_boundary_dof = dof_handler.n_dofs() + 1;
        if (locally_relevant_boundary_dofs.n_elements() > 0)
          first_local_boundary_dof =
            locally_relevant_boundary_dofs.nth_index_in_set(0);

        // first boundary dof is minimum of all
        const unsigned int first_boundary_dof =
          dealii::Utilities::MPI::min(first_local_boundary_dof,
                                      mpi_communicator);

        /*
         * This constrains only the first dof on the first processor. We set it
         * to zero. Note that setting a point value may be problematic for an
         * H1-Function but in parallel adding mean value constraints should not
         * be done.
         */
        if (first_boundary_dof == first_local_boundary_dof)
          constraints.add_line(first_boundary_dof);
      }

    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      Utilities::MPI::all_gather(mpi_communicator,
                                 dof_handler.n_locally_owned_dofs()),
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }



  void
    QStd::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    QGauss<3> quadrature_formula(fe.degree + 1);
    QGauss<2> face_quadrature_formula(fe.degree + 1);

    FEValues<3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<3> fe_face_values(fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_normal_vectors | update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    /*
     * Matrix coefficient and vector to store the values.
     */
    const EquationData::Diffusion_A matrix_coeff(parameter_filename);
    std::vector<Tensor<2, 3>>       matrix_coeff_values(n_q_points);

    /*
     * Reaction rate.
     */
    const EquationData::ReactionRate reaction_rate;
    std::vector<double>              reaction_rate_values(n_q_points);

    /*
     * Right hand side and vector to store the values.
     */
    const EquationData::RightHandSideParsed right_hand_side(
      parameter_filename, /* n_components */ 1);
    std::vector<double> rhs_values(n_q_points);

    /*
     * Neumann BCs and vector to store the values.
     */
    const EquationData::Boundary_A_grad_u neumann_bc;
    std::vector<Tensor<1, 3>>             neumann_values(n_face_q_points);

    /*
     * Integration over cells.
     */
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell_matrix = 0;
            cell_rhs    = 0;

            fe_values.reinit(cell);

            // Now actually fill with values.
            matrix_coeff.value_list(fe_values.get_quadrature_points(),
                                    matrix_coeff_values);
            reaction_rate.value_list(fe_values.get_quadrature_points(),
                                     reaction_rate_values);
            right_hand_side.value_list(fe_values.get_quadrature_points(),
                                       rhs_values);

            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          (fe_values.shape_grad(i, q_index) *
                             matrix_coeff_values[q_index] *
                             fe_values.shape_grad(j, q_index) +
                           fe_values.shape_value(i, q_index) *
                             reaction_rate_values[q_index] *
                             fe_values.shape_value(j, q_index)) *
                          fe_values.JxW(q_index);
                      } // end ++j

                    cell_rhs(i) += fe_values.shape_value(i, q_index) *
                                   rhs_values[q_index] * fe_values.JxW(q_index);
                  } // end ++i
              }     // end ++q_index

            if (parameters.is_pure_neumann)
              {
                /*
                 * Boundary integral for Neumann values for odd boundary_id.
                 */
                for (unsigned int face_number = 0;
                     face_number < GeometryInfo<3>::faces_per_cell;
                     ++face_number)
                  {
                    if (cell->face(face_number)->at_boundary())
                      {
                        fe_face_values.reinit(cell, face_number);

                        /*
                         * Fill in values at this particular face.
                         */
                        neumann_bc.value_list(
                          fe_face_values.get_quadrature_points(),
                          neumann_values);

                        for (unsigned int q_face_point = 0;
                             q_face_point < n_face_q_points;
                             ++q_face_point)
                          {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                cell_rhs(i) +=
                                  (neumann_values[q_face_point] *
                                   fe_face_values.normal_vector(
                                     q_face_point) // g(x_q)
                                   ) *
                                  fe_face_values.shape_value(
                                    i, q_face_point) // phi_i(x_q)
                                  * fe_face_values.JxW(q_face_point); // dS
                              }                                       // end ++i
                          } // end ++q_face_point
                      }     // end if
                  }         // end ++face_number
              }


            // get global indices
            cell->get_dof_indices(local_dof_indices);
            /*
             * Now add the cell matrix and rhs to the right spots
             * in the global matrix and global rhs. Constraints will
             * be taken care of later.
             */
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          } // end if (cell->is_locally_owned())
      }     // end ++cell

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  void
    QStd::solve_direct()
  {
    TimerOutput::Scope t(computing_timer,
                         "parallel sparse direct solver (MUMPS)");

    TrilinosWrappers::MPI::Vector completely_distributed_solution(
      locally_owned_dofs, mpi_communicator);

    SolverControl                  solver_control;
    TrilinosWrappers::SolverDirect solver(solver_control);
    solver.initialize(system_matrix);

    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    pcout << "   Solved in with parallel sparse direct solver (MUMPS)."
          << std::endl;

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
  }



  void
    QStd::solve_iterative()
  {
    TimerOutput::Scope t(computing_timer, "iterative solver");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

#ifdef USE_PETSC_LA
    LA::SolverCG solver(solver_control, mpi_communicator);
#else
    LA::SolverCG solver(solver_control);
#endif

    LA::MPI::PreconditionAMG                 preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    /* Trilinos defaults are good */
#endif

    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }


  void
    QStd::transfer_solution()
  {
    TimerOutput::Scope t(computing_timer, "solution transfer");

    /*
     * Refine everything.
     */
    {
      for (typename Triangulation<3>::active_cell_iterator cell =
             triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        if (cell->is_locally_owned())
          cell->set_refine_flag();
    }

    /*
     * Prepare the triangulation for refinement.
     */
    triangulation.prepare_coarsening_and_refinement();

    /*
     * Prepare the refinement in the transfer object,
     * locally_relevant_old_solution is the source.
     */
    parallel::distributed::SolutionTransfer<3, LA::MPI::Vector>
      solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(
      locally_relevant_solution);

    /*
     * Now actually refine the mesh
     */
    triangulation.execute_coarsening_and_refinement();

    { /*
       * Setup new dofs and constraints.
       */
      dof_handler.distribute_dofs(fe);

      locally_owned_dofs = dof_handler.locally_owned_dofs();
      locally_relevant_dofs.clear();
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      locally_relevant_solution.reinit(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);

      system_rhs.reinit(locally_owned_dofs, mpi_communicator);

      constraints.clear();
      constraints.reinit(locally_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      if (!parameters.is_pure_neumann)
        {
          // Set up Dirichlet boundary conditions.
          const EquationData::BoundaryValues_u boundary_u;
          for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
            {
              VectorTools::interpolate_boundary_values(
                dof_handler,
                /*boundary id*/ i, // only even boundary id
                boundary_u,
                constraints);
            }
        }
    }


    /*
     * New locally_owned_solution from new dofs.
     */
    TrilinosWrappers::MPI::Vector locally_owned_solution;
    locally_owned_solution.reinit(locally_owned_dofs, mpi_communicator);

    /*
     * Now interpolate to new mesh.
     */
    solution_transfer.interpolate(locally_owned_solution);

    /*
     * Take care of constraints.
     */
    constraints.distribute(locally_owned_solution);

    locally_relevant_solution = locally_owned_solution;
  }


  void
    QStd::output_results() const
  {
    std::vector<std::string> solution_names(1, "u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        1, DataComponentInterpretation::component_is_scalar);

    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<3>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain_id");

    // Postprocess
    std::unique_ptr<Q_PostProcessor> postprocessor(
      new Q_PostProcessor(parameter_filename));
    data_out.add_data_vector(locally_relevant_solution, *postprocessor);

    data_out.build_patches();

    std::string filename(parameters.filename_output);
    filename += "_n_refine-" + Utilities::int_to_string(parameters.n_refine, 2);
    filename +=
      "." +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4);
    filename += ".vtu";

    std::ofstream output(parameters.dirname_output + "/" + filename);
    data_out.write_vtu(output);

    // pvtu-record for all local outputs
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> local_filenames(
          Utilities::MPI::n_mpi_processes(mpi_communicator),
          parameters.filename_output);
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          {
            local_filenames[i] +=
              "_n_refine-" + Utilities::int_to_string(parameters.n_refine, 2) +
              "." + Utilities::int_to_string(i, 4) + ".vtu";
          }

        std::string master_file =
          parameters.filename_output + "_n_refine-" +
          Utilities::int_to_string(parameters.n_refine, 2) + ".pvtu";
        std::ofstream master_output(parameters.dirname_output + "/" +
                                    master_file);
        data_out.write_pvtu_record(master_output, local_filenames);
      }
  }



  void
    QStd::run()
  {
    if (parameters.compute_solution == false)
      {
        deallog << "Run of standard problem is explicitly disabled in "
                   "parameter file. "
                << std::endl;
        return;
      }

    pcout << std::endl
          << "===========================================" << std::endl
          << "Solving >> Q1 STANDARD << problem in 3D." << std::endl;

    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    make_grid();

    setup_system();

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    assemble_system();

    if (parameters.use_direct_solver)
      solve_direct(); // SparseDirectMUMPS
    else
      {
        solve_iterative();
      }

    const int n_transfer = parameters.transfer_to_level - parameters.n_refine;
    if (n_transfer > 0)
      {
        pcout << std::endl
              << "INFO: Transfer to finer grid by   " << n_transfer
              << "   levels to global refinement level   "
              << parameters.transfer_to_level << std::endl
              << std::endl;

        for (int i = 0; i < n_transfer; ++i)
          transfer_solution();
      }
    else
      {
        pcout
          << "INFO: Transfer to coarser or same grid requested. This is being ignored so that the solution is not being transferred at all."
          << std::endl;
      }


    {
      TimerOutput::Scope t(computing_timer, "vtu output");
      try
        {
          Tools::create_data_directory(parameters.dirname_output);
        }
      catch (std::runtime_error &e)
        {
          // No exception handling here.
        }
      output_results();
    }

    if (parameters.verbose)
      {
        computing_timer.print_summary();
        computing_timer.reset();
      }

    pcout << std::endl
          << "===========================================" << std::endl;
  }

} // end namespace Q
