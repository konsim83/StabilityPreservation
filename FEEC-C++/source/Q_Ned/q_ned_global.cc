#include <Q_Ned/q_ned_global.h>

namespace QNed
{
  using namespace dealii;

  QNedMultiscale::QNedMultiscale(QNed::ParametersMs &parameters_,
                                 const std::string & parameter_filename_)
    : mpi_communicator(MPI_COMM_WORLD)
    , parameters(parameters_)
    , parameter_filename(parameter_filename_)
    , triangulation(mpi_communicator,
                    typename Triangulation<3>::MeshSmoothing(
                      Triangulation<3>::smoothing_on_refinement |
                      Triangulation<3>::smoothing_on_coarsening))
    , fe(FE_Q<3>(1), 1, FE_Nedelec<3>(0), 1)
    , dof_handler(triangulation)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , cell_basis_map()
  {
    if ((parameters.renumber_dofs) && (parameters.use_exact_solution))
      throw std::runtime_error(
        "When using the exact solution dof renumbering must be disabled in "
        "parameter file.");
  }

  QNedMultiscale::~QNedMultiscale()
  {
    system_matrix.clear();
    constraints.clear();
    dof_handler.clear();
  }

  void
    QNedMultiscale::setup_grid()
  {
    TimerOutput::Scope t(computing_timer, "coarse mesh generation");

    GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);

    triangulation.refine_global(parameters.n_refine_global);
  }

  void
    QNedMultiscale::initialize_and_compute_basis()
  {
    TimerOutput::Scope t(computing_timer,
                         "Q1-Nedelec basis initialization and computation");

    typename Triangulation<3>::active_cell_iterator cell = dof_handler
                                                             .begin_active(),
                                                    endc = dof_handler.end();
    first_cell =
      cell->id(); // only to identify first cell (for setting output flag etc)

    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            QNedBasis current_cell_problem(
              parameters,
              parameter_filename,
              cell,
              first_cell,
              triangulation.locally_owned_subdomain(),
              mpi_communicator);
            CellId current_cell_id(cell->id());

            std::pair<typename std::map<CellId, QNedBasis>::iterator, bool>
              result;
            result = cell_basis_map.insert(
              std::make_pair(cell->id(), current_cell_problem));

            Assert(result.second,
                   ExcMessage(
                     "Insertion of local basis problem into std::map failed. "
                     "Problem with copy constructor?"));
          }
      } // end ++cell

    /*
     * Now each node possesses a set of basis objects.
     * We need to compute them on each node and do so in
     * a locally threaded way.
     */
    typename std::map<CellId, QNedBasis>::iterator it_basis =
                                                     cell_basis_map.begin(),
                                                   it_endbasis =
                                                     cell_basis_map.end();
    for (; it_basis != it_endbasis; ++it_basis)
      {
        (it_basis->second).run();
      }
  }

  void
    QNedMultiscale::setup_system_matrix()
  {
    TimerOutput::Scope t(computing_timer, "system and constraint setup");

    dof_handler.distribute_dofs(fe);

    if (parameters.renumber_dofs)
      {
        DoFRenumbering::Cuthill_McKee(dof_handler);
      }

    DoFRenumbering::block_wise(dof_handler);

    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);
    const unsigned int n_sigma = dofs_per_block[0], n_u = dofs_per_block[1];

    pcout << "Number of active cells: " << triangulation.n_global_active_cells()
          << std::endl
          << "Total number of cells: " << triangulation.n_cells() << " (on "
          << triangulation.n_levels() << " levels)" << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
          << n_sigma << '+' << n_u << ')' << std::endl;

    owned_partitioning.resize(2);
    owned_partitioning[0] =
      dof_handler.locally_owned_dofs().get_view(0, n_sigma);
    owned_partitioning[1] =
      dof_handler.locally_owned_dofs().get_view(n_sigma, n_sigma + n_u);

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_sigma);
    relevant_partitioning[1] =
      locally_relevant_dofs.get_view(n_sigma, n_sigma + n_u);

    setup_constraints();

    {
      // Allocate memory
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        Utilities::MPI::all_gather(mpi_communicator,
                                   dof_handler.locally_owned_dofs()),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.clear();
      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

      //		preconditioner_matrix.clear();
      //		preconditioner_matrix.reinit(owned_partitioning, dsp,
      // mpi_communicator);
    }

    locally_relevant_solution.reinit(owned_partitioning,
                                     relevant_partitioning,
                                     mpi_communicator);

    system_rhs.reinit(owned_partitioning, mpi_communicator);
  }

  void
    QNedMultiscale::setup_constraints()
  {
    // set constraints (first hanging nodes, then flux)
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // boundary values
    std::unique_ptr<Function<3>>   boundary_values;
    std::unique_ptr<Function<3>>   sigma;
    EquationData::ExactSolutionLin u_tensor(parameter_filename);
    std::unique_ptr<Function<3>>   u;
    if (parameters.use_exact_solution)
      {
        sigma.reset(
          new EquationData::ExactSolutionLin_B_div(parameter_filename));
        u.reset(new VectorFunctionFromTensorFunction<3>(u_tensor));

        boundary_values.reset(
          new ShapeFun::ShapeFunctionConcatinateVector<3>(*sigma, *u));
      }
    else
      boundary_values.reset(new Functions::ZeroFunction<3>(4));

    FEValuesExtractors::Scalar q1(0);
    ComponentMask              q1_mask = fe.component_mask(q1);
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
      {
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 /*boundary id*/ i,
                                                 *boundary_values,
                                                 constraints,
                                                 q1_mask);
        VectorTools::project_boundary_values_curl_conforming_l2(
          dof_handler,
          /*first vector component */ 1,
          *boundary_values,
          /*boundary id*/ i,
          constraints);
      }

    constraints.close();
  }

  void
    QNedMultiscale::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "multiscale assembly");

    system_matrix = 0;
    system_rhs    = 0;

    //	QGauss<2> 	face_quadrature_formula(3);

    // Get relevant quantities to be updated from finite element
    //	FEFaceValues<3> fe_face_values (fe, face_quadrature_formula,
    //									  update_values    | update_normal_vectors
    //| 									  update_quadrature_points  | update_JxW_values);

    // Define some abbreviations
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    //	const unsigned int   n_face_q_points =
    // face_quadrature_formula.size();

    // Declare local contributions and reserve memory
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // ------------------------------------------------------------------
    // loop over cells
    typename DoFHandler<3>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            typename std::map<CellId, QNedBasis>::iterator it_basis =
              cell_basis_map.find(cell->id());

            local_matrix = 0;
            local_rhs    = 0;

            local_matrix = (it_basis->second).get_global_element_matrix();
            local_rhs    = (it_basis->second).get_global_element_rhs();

            // line integral over boundary faces for for natural
            // conditions on u
            //			for (unsigned int face_n=0;
            //						 face_n<GeometryInfo<3>::faces_per_cell;
            //						 ++face_n)
            //			{
            //				if (cell->at_boundary(face_n)
            //			//					&&
            // cell->face(face_n)->boundary_id()!=0
            ///*
            // Select only certain faces. */
            //			//					&&
            // cell->face(face_n)->boundary_id()!=2
            ///*
            // Select only certain faces. */
            //						)
            //				{
            //					fe_face_values.reinit (cell,
            // face_n);
            //
            //					boundary_values_u.value_list
            //(fe_face_values.get_quadrature_points(),
            //						  boundary_values_u_values);
            //
            //					for (unsigned int q=0; q<n_face_q_points;
            //++q) 						for (unsigned int i=0;
            // i<dofs_per_cell; ++i) local_rhs(i) +=
            //-(fe_face_values[flux].value (i, q)
            //* fe_face_values.normal_vector(q)
            //*
            // boundary_values_u_values[q] * fe_face_values.JxW(q));
            //				}
            //			}

            // Add to global matrix, include constraints
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(
              local_matrix,
              local_rhs,
              local_dof_indices,
              system_matrix,
              system_rhs,
              /* use inhomogeneities for rhs */ true);
          }
      } // end for ++cell

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  void
    QNedMultiscale::solve_direct()
  {
    TimerOutput::Scope t(computing_timer, " direct solver (MUMPS)");

    throw std::runtime_error(
      "Solver not implemented: MUMPS does not work on "
      "TrilinosWrappers::MPI::BlockSparseMatrix classes.");
  }

  void
    QNedMultiscale::solve_iterative()
  {
    inner_schur_preconditioner =
      std::make_shared<typename LinearSolvers::InnerPreconditioner<3>::type>();

    typename LinearSolvers::InnerPreconditioner<3>::type::AdditionalData data;
#ifdef USE_PETSC_LA
//	data.symmetric_operator = true; // Only for AMG
#endif

    inner_schur_preconditioner->initialize(system_matrix.block(0, 0), data);

    const LinearSolvers::InverseMatrix<
      LA::MPI::SparseMatrix,
      typename LinearSolvers::InnerPreconditioner<3>::type>
      block_inverse(system_matrix.block(0, 0), *inner_schur_preconditioner);

    // Vector for solution
    LA::MPI::BlockVector distributed_solution(owned_partitioning,
                                              mpi_communicator);

    // tmp of size block(0)
    LA::MPI::Vector tmp(owned_partitioning[0], mpi_communicator);

    // Set up Schur complement
    LinearSolvers::SchurComplementMPI<
      LA::MPI::BlockSparseMatrix,
      LA::MPI::Vector,
      typename LinearSolvers::InnerPreconditioner<3>::type>
      schur_complement(system_matrix,
                       block_inverse,
                       owned_partitioning,
                       mpi_communicator);

    // Compute schur_rhs = -g + C*A^{-1}*f
    LA::MPI::Vector schur_rhs(owned_partitioning[1],
                              //								relevant_partitioning[1],
                              mpi_communicator);
    block_inverse.vmult(tmp, system_rhs.block(0));
    system_matrix.block(1, 0).vmult(schur_rhs, tmp);
    schur_rhs -= system_rhs.block(1);
    {
      TimerOutput::Scope t(computing_timer, "Schur complement solver (for u)");

      // Set Solver parameters for solving for u
      SolverControl             solver_control(system_matrix.m(),
                                   1e-6 * schur_rhs.l2_norm());
      SolverCG<LA::MPI::Vector> schur_solver(solver_control);

      //		PreconditionIdentity preconditioner;

      /*
       * Precondition the Schur complement with
       * the approximate inverse of the
       * Schur complement.
       */
      //		LinearSolvers::ApproximateInverseMatrix<LinearSolvers::SchurComplementMPI<LA::MPI::BlockSparseMatrix,
      //																					LA::MPI::Vector,
      //																					typename
      // LinearSolvers::InnerPreconditioner<3>::type>,
      // PreconditionIdentity>
      // preconditioner (schur_complement,
      //												PreconditionIdentity()
      //);

      /*
       * Precondition the Schur complement with
       * the (approximate) inverse of an approximate
       * Schur complement.
       */
      LinearSolvers::ApproximateSchurComplementMPI<LA::MPI::BlockSparseMatrix,
                                                   LA::MPI::Vector,
                                                   LA::MPI::PreconditionILU>
        approx_schur(system_matrix, owned_partitioning, mpi_communicator);

      LinearSolvers::ApproximateInverseMatrix<
        LinearSolvers::ApproximateSchurComplementMPI<LA::MPI::BlockSparseMatrix,
                                                     LA::MPI::Vector,
                                                     LA::MPI::PreconditionILU>,
        PreconditionIdentity>
        preconditioner(approx_schur,
                       PreconditionIdentity(),
#ifdef DEBUG
                       /* n_iter */ 1000);
#else
                       /* n_iter */ 14);
#endif

      /*
       * Precondition the Schur complement with a preconditioner of
       * block(1,1).
       */
      //		LA::MPI::PreconditionAMG preconditioner;
      //		preconditioner.initialize(system_matrix.block(1, 1),
      // data);

      schur_solver.solve(schur_complement,
                         distributed_solution.block(1),
                         schur_rhs,
                         preconditioner);

      pcout << "   Iterative Schur complement solver converged in "
            << solver_control.last_step() << " iterations." << std::endl;

      constraints.distribute(distributed_solution);
    }

    {
      TimerOutput::Scope t(computing_timer, "outer CG solver (for sigma)");

      //	SolverControl                    outer_solver_control;
      //	PETScWrappers::SparseDirectMUMPS
      // outer_solver(outer_solver_control,
      // mpi_communicator); 	outer_solver.set_symmetric_mode(true);

      // use computed u to solve for sigma
      system_matrix.block(0, 1).vmult(tmp, distributed_solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);

      // Solve for sigma
      //	outer_solver.solve(system_matrix.block(0,0),
      // distributed_solution.block(0), tmp);
      block_inverse.vmult(distributed_solution.block(0), tmp);

      pcout << "   Outer solver completed." << std::endl;

      constraints.distribute(distributed_solution);
    }

    locally_relevant_solution = distributed_solution;
  }

  void
    QNedMultiscale::send_global_weights_to_cell()
  {
    // For each cell we get dofs_per_cell values
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // active cell iterator
    typename DoFHandler<3>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            std::vector<double> extracted_weights(dofs_per_cell, 0);
            locally_relevant_solution.extract_subvector_to(local_dof_indices,
                                                           extracted_weights);

            typename std::map<CellId, QNedBasis>::iterator it_basis =
              cell_basis_map.find(cell->id());
            (it_basis->second).set_global_weights(extracted_weights);
          }
      } // end ++cell
  }

  void
    QNedMultiscale::write_exact_solution()
  {
    locally_relevant_exact_solution.reinit(owned_partitioning,
                                           relevant_partitioning,
                                           mpi_communicator);

    { // write sigma
      // Quadrature used for projection
      QGauss<3> quad_rule(3);

      // Setup function
      EquationData::ExactSolutionLin_B_div exact_sigma(parameter_filename);

      DoFHandler<3> dof_handler_fake(triangulation);
      dof_handler_fake.distribute_dofs(fe.base_element(0));

      AffineConstraints<double> constraints_fake;
      constraints_fake.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_fake,
                                              constraints_fake);
      constraints_fake.close();

      MyVectorTools::project_on_fe_space(dof_handler_fake,
                                         constraints_fake,
                                         quad_rule,
                                         exact_sigma,
                                         locally_relevant_exact_solution.block(
                                           0),
                                         mpi_communicator);

      dof_handler_fake.clear();
    }

    { // write u
      // Quadrature used for projection
      QGauss<3> quad_rule(3);

      // Setup function
      EquationData::ExactSolutionLin exact_u(parameter_filename);

      DoFHandler<3> dof_handler_fake(triangulation);
      dof_handler_fake.distribute_dofs(fe.base_element(1));

      AffineConstraints<double> constraints_fake;
      constraints_fake.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_fake,
                                              constraints_fake);
      constraints_fake.close();

      MyVectorTools::project_on_fe_space(dof_handler_fake,
                                         constraints_fake,
                                         quad_rule,
                                         exact_u,
                                         locally_relevant_exact_solution.block(
                                           1),
                                         mpi_communicator);

      dof_handler_fake.clear();
    }
  }

  std::vector<std::string>
    QNedMultiscale::collect_filenames_on_mpi_process()
  {
    std::vector<std::string> filename_list;

    typename std::map<CellId, QNedBasis>::iterator it_basis =
                                                     cell_basis_map.begin(),
                                                   it_endbasis =
                                                     cell_basis_map.end();

    for (; it_basis != it_endbasis; ++it_basis)
      {
        filename_list.push_back((it_basis->second).get_filename_global());
      }

    return filename_list;
  }

  void
    QNedMultiscale::output_results()
  {
    // write local fine solution
    typename std::map<CellId, QNedBasis>::iterator it_basis =
                                                     cell_basis_map.begin(),
                                                   it_endbasis =
                                                     cell_basis_map.end();

    for (; it_basis != it_endbasis; ++it_basis)
      {
        (it_basis->second).output_global_solution_in_cell();
      }

    // Gather local filenames
    std::vector<std::vector<std::string>> filename_list_list =
      Utilities::MPI::gather(mpi_communicator,
                             collect_filenames_on_mpi_process(),
                             /* root_process = */ 0);

    std::vector<std::string> filenames_on_cell;
    for (unsigned int i = 0; i < filename_list_list.size(); ++i)
      for (unsigned int j = 0; j < filename_list_list[i].size(); ++j)
        filenames_on_cell.push_back(filename_list_list[i][j]);

    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names(1, "sigma");
    solution_names.emplace_back("u");
    solution_names.emplace_back("u");
    solution_names.emplace_back("u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        1, DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<3>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();

    data_out.add_data_vector(subdomain, "subdomain_id");

    // Postprocess
    std::unique_ptr<QNed_PostProcessor> postprocessor(
      new QNed_PostProcessor(parameter_filename,
                             parameters.use_exact_solution));
    data_out.add_data_vector(locally_relevant_solution, *postprocessor);

    // Postprocess exact solution if needed
    std::vector<std::string> solution_names_exact(1, "exact_sigma");
    solution_names_exact.emplace_back("exact_u");
    solution_names_exact.emplace_back("exact_u");
    solution_names_exact.emplace_back("exact_u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_exact(
        1, DataComponentInterpretation::component_is_scalar);
    data_component_interpretation_exact.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation_exact.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation_exact.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

    if (parameters.use_exact_solution)
      {
        data_out.add_data_vector(locally_relevant_exact_solution,
                                 solution_names_exact,
                                 DataOut<3>::type_dof_data,
                                 data_component_interpretation_exact);
      }

    std::unique_ptr<QNed_PostProcessor> postprocessor_exact;
    if (parameters.use_exact_solution)
      {
        postprocessor_exact.reset(new QNed_PostProcessor(
          parameter_filename, parameters.use_exact_solution, "exact_"));
        data_out.add_data_vector(locally_relevant_exact_solution,
                                 *postprocessor_exact);
      } // end if (parameters.use_exact_solution)

    data_out.build_patches();

    std::string filename(parameters.filename_output);
    filename +=
      "_n_refine-" + Utilities::int_to_string(parameters.n_refine_global, 2);
    filename +=
      "." +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4);
    filename += ".vtu";

    std::ofstream output(parameters.dirname_output + "/" + filename);
    data_out.write_vtu(output);

    // pvtu-record for all local coarse outputs
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
              "_n_refine-" +
              Utilities::int_to_string(parameters.n_refine_global, 2) + "." +
              Utilities::int_to_string(i, 4) + ".vtu";
          }

        std::string master_file =
          parameters.filename_output + "_n_refine-" +
          Utilities::int_to_string(parameters.n_refine_global, 2) + ".pvtu";
        std::ofstream master_output(parameters.dirname_output + "/" +
                                    master_file);
        data_out.write_pvtu_record(master_output, local_filenames);
      }

    // pvtu-record for all local fine outputs
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::string filename_master = parameters.filename_output;
        filename_master += "_fine";
        filename_master +=
          "_refine-" + Utilities::int_to_string(parameters.n_refine_global, 2) +
          "-" + Utilities::int_to_string(parameters.n_refine_local, 2) +
          ".pvtu";

        std::ofstream master_output(parameters.dirname_output + "/" +
                                    filename_master);
        data_out.write_pvtu_record(master_output, filenames_on_cell);
      }
  }

  void
    QNedMultiscale::run()
  {
    if (parameters.compute_solution == false)
      {
        deallog << "Run of multiscale problem is explicitly disabled in "
                   "parameter file. "
                << std::endl;
        return;
      }

    pcout << std::endl
          << "===========================================" << std::endl
          << "Solving >> modified Q1-Ned MULTISCALE << problem in 3D."
          << std::endl;

#ifdef USE_PETSC_LA
    pcout << "Running multiscale algorithm using PETSc." << std::endl;
#else
    pcout << "Running multiscale algorithm using Trilinos." << std::endl;
#endif

    setup_grid();

    setup_system_matrix();

    initialize_and_compute_basis();

    setup_constraints();

    assemble_system();

    if (parameters.use_direct_solver)
      solve_direct(); // SparseDirectMUMPS
    else
      {
        solve_iterative(); // Schur complement for A
      }

    send_global_weights_to_cell();

    if (parameters.use_exact_solution)
      {
        TimerOutput::Scope t(computing_timer, "projection of exact solution");
        write_exact_solution();
      }

    {
      TimerOutput::Scope t(computing_timer, "vtu output fine");
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

} // end namespace QNed
