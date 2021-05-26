#include <Ned_RT/ned_rt_basis.h>

namespace NedRT
{
  using namespace dealii;

  NedRTBasis::NedRTBasis(
    const NedRT::ParametersMs &                      parameters_ms,
    const std::string &                              parameter_filename_,
    typename Triangulation<3>::active_cell_iterator &global_cell,
    CellId                                           first_cell,
    unsigned int                                     local_subdomain,
    MPI_Comm                                         mpi_communicator)
    : mpi_communicator(mpi_communicator)
    , parameters(parameters_ms)
    , parameter_filename(parameter_filename_)
    , triangulation()
    , fe(FE_Nedelec<3>(parameters.degree),
         1,
         FE_RaviartThomas<3>(parameters.degree),
         1)
    , dof_handler(triangulation)
    , constraints_curl_v(GeometryInfo<3>::lines_per_cell)
    , constraints_div_v(GeometryInfo<3>::faces_per_cell)
    , sparsity_pattern()
    , basis_curl_v(GeometryInfo<3>::lines_per_cell)
    , basis_div_v(GeometryInfo<3>::faces_per_cell)
    , system_rhs_curl_v(GeometryInfo<3>::lines_per_cell)
    , system_rhs_div_v(GeometryInfo<3>::faces_per_cell)
    , global_element_matrix(fe.dofs_per_cell, fe.dofs_per_cell)
    , global_element_rhs(fe.dofs_per_cell)
    , global_weights(fe.dofs_per_cell, 0)
    , global_cell_id(global_cell->id())
    , first_cell(first_cell)
    , global_cell_it(global_cell)
    , local_subdomain(local_subdomain)
    , volume_measure(0)
    , face_measure(GeometryInfo<3>::faces_per_cell, 0)
    , edge_measure(GeometryInfo<3>::lines_per_cell, 0)
    , corner_points(GeometryInfo<3>::vertices_per_cell, Point<3>())
    , length_system_basis(GeometryInfo<3>::lines_per_cell +
                          GeometryInfo<3>::faces_per_cell)
    , is_built_global_element_matrix(false)
    , is_set_global_weights(false)
    , is_set_cell_data(false)
    , is_copyable(true)
  {
    if ((parameters.renumber_dofs) && (parameters.use_exact_solution))
      throw std::runtime_error(
        "When using the exact solution dof renumbering must be disabled in "
        "parameter file.");

    for (unsigned int vertex_n = 0;
         vertex_n < GeometryInfo<3>::vertices_per_cell;
         ++vertex_n)
      {
        corner_points.at(vertex_n) = global_cell_it->vertex(vertex_n);
      }

    volume_measure = global_cell_it->measure();

    for (unsigned int j_face = 0; j_face < GeometryInfo<3>::faces_per_cell;
         ++j_face)
      {
        face_measure.at(j_face) = global_cell_it->face(j_face)->measure();
      }

    for (unsigned int j_egde = 0; j_egde < GeometryInfo<3>::lines_per_cell;
         ++j_egde)
      {
        edge_measure.at(j_egde) = global_cell_it->line(j_egde)->measure();
      }

    is_set_cell_data = true;
  }

  NedRTBasis::NedRTBasis(const NedRTBasis &other)
    : mpi_communicator(other.mpi_communicator)
    , parameters(other.parameters)
    , parameter_filename(other.parameter_filename)
    , triangulation()
    , // must be constructed deliberately, but is empty on
      // copying anyway
    fe(FE_Nedelec<3>(parameters.degree),
       1,
       FE_RaviartThomas<3>(parameters.degree),
       1)
    , dof_handler(triangulation)
    , constraints_curl_v(other.constraints_curl_v)
    , constraints_div_v(other.constraints_div_v)
    //    , sparsity_pattern(
    //        other.sparsity_pattern) // only possible if object is empty
    //    , assembled_matrix(
    //        other.assembled_matrix)          // only possible if object is
    //        empty
    //    , system_matrix(other.system_matrix) // only possible if object is
    //    empty
    , basis_curl_v(other.basis_curl_v)
    , basis_div_v(other.basis_div_v)
    , system_rhs_curl_v(other.system_rhs_curl_v)
    , system_rhs_div_v(other.system_rhs_div_v)
    , global_rhs(other.global_rhs)
    , global_element_matrix(other.global_element_matrix)
    , global_element_rhs(other.global_element_rhs)
    , global_weights(other.global_weights)
    , global_solution(other.global_solution)
    , exact_solution_in_cell(other.exact_solution_in_cell)
    , inner_schur_preconditioner(other.inner_schur_preconditioner)
    , global_cell_id(other.global_cell_id)
    , first_cell(other.first_cell)
    , global_cell_it(other.global_cell_it)
    , local_subdomain(other.local_subdomain)
    , volume_measure(other.volume_measure)
    , face_measure(other.face_measure)
    , edge_measure(other.edge_measure)
    , corner_points(other.corner_points)
    , length_system_basis(other.length_system_basis)
    , is_built_global_element_matrix(other.is_built_global_element_matrix)
    , is_set_global_weights(other.is_set_global_weights)
    , is_set_cell_data(other.is_set_cell_data)
    , is_copyable(other.is_copyable)
  {
    global_cell_id = global_cell_it->id();

    for (unsigned int vertex_n = 0;
         vertex_n < GeometryInfo<3>::vertices_per_cell;
         ++vertex_n)
      {
        corner_points.at(vertex_n) = global_cell_it->vertex(vertex_n);
      }

    volume_measure = global_cell_it->measure();

    for (unsigned int j_face = 0; j_face < GeometryInfo<3>::faces_per_cell;
         ++j_face)
      {
        face_measure.at(j_face) = global_cell_it->face(j_face)->measure();
      }

    for (unsigned int j_egde = 0; j_egde < GeometryInfo<3>::lines_per_cell;
         ++j_egde)
      {
        edge_measure.at(j_egde) = global_cell_it->line(j_egde)->measure();
      }

    is_set_cell_data = true;
  }

  NedRTBasis::~NedRTBasis()
  {
    system_matrix.clear();

    for (unsigned int n_basis = 0; n_basis < basis_curl_v.size(); ++n_basis)
      {
        constraints_curl_v[n_basis].clear();
      }

    for (unsigned int n_basis = 0; n_basis < basis_div_v.size(); ++n_basis)
      {
        constraints_div_v[n_basis].clear();
      }

    dof_handler.clear();
  }

  void
    NedRTBasis::setup_grid()
  {
    Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

    GridGenerator::general_cell(triangulation,
                                corner_points,
                                /* colorize faces */ false);

    triangulation.refine_global(parameters.n_refine_local);

    is_copyable = false;
  }

  void
    NedRTBasis::setup_system_matrix()
  {
    dof_handler.distribute_dofs(fe);

    if (parameters.renumber_dofs)
      {
        DoFRenumbering::Cuthill_McKee(dof_handler);
      }

    DoFRenumbering::block_wise(dof_handler);

    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);
    const unsigned int n_sigma = dofs_per_block[0], n_u = dofs_per_block[1];

    if (parameters.verbose)
      {
        std::cout << "Number of active cells: "
                  << triangulation.n_active_cells() << std::endl
                  << "Total number of cells: " << triangulation.n_cells()
                  << std::endl
                  << "Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << n_sigma << '+' << n_u << ')' << std::endl;
      }

    {
      // Allocate memory
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

      DoFTools::make_sparsity_pattern(dof_handler, dsp);

      // Initialize the system matrix for global assembly
      sparsity_pattern.copy_from(dsp);
    }

    assembled_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    global_solution.reinit(dofs_per_block);

    global_rhs.reinit(dofs_per_block);
  }

  void
    NedRTBasis::setup_basis_dofs_curl()
  {
    Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

    Timer timer;

    if (parameters.verbose)
      {
        std::cout << "	Setting up dofs for H(curl) part.....";

        timer.restart();
      }

    ShapeFun::BasisNedelec<3>  std_shape_function_Ned(global_cell_it,
                                                     /* degree */ 0);
    Functions::ZeroFunction<3> zero_fun_vector(3);
    ShapeFun::ShapeFunctionConcatinateVector<3> std_shape_function(
      std_shape_function_Ned, zero_fun_vector);

    ShapeFun::BasisNedelecCurl<3> std_shape_function_Ned_curl(global_cell_it,
                                                              /* degree */ 0);

    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);

    // set constraints (first hanging nodes, then boundary conditions)
    for (unsigned int n_basis = 0; n_basis < basis_curl_v.size(); ++n_basis)
      {
        std_shape_function_Ned.set_index(n_basis);
        std_shape_function_Ned_curl.set_index(n_basis);

        constraints_curl_v[n_basis].clear();

        DoFTools::make_hanging_node_constraints(dof_handler,
                                                constraints_curl_v[n_basis]);

        VectorTools::project_boundary_values_curl_conforming_l2(
          dof_handler,
          /*first vector component */ 0,
          std_shape_function, // This is important!!!
                              //          Functions::ZeroFunction<3>(6),
          /*boundary id*/ 0,
          constraints_curl_v[n_basis]);
        VectorTools::project_boundary_values_div_conforming(
          dof_handler,
          /*first vector component */ 3,
          //          std_shape_function_Ned_curl, // This is important only if
          //          full rhs not used!!!
          Functions::ZeroFunction<3>(3),
          /*boundary id*/ 0,
          constraints_curl_v[n_basis]);

        constraints_curl_v[n_basis].close();
      }

    for (unsigned int n_basis = 0; n_basis < basis_curl_v.size(); ++n_basis)
      {
        basis_curl_v[n_basis].reinit(dofs_per_block);
        system_rhs_curl_v[n_basis].reinit(dofs_per_block);
      }

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

  void
    NedRTBasis::setup_basis_dofs_div()
  {
    Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

    Timer timer;

    if (parameters.verbose)
      {
        std::cout << "	Setting up dofs for H(div) part.....";

        timer.restart();
      }

    //    ShapeFun::ShapeFunctionVector<3>
    //    std_shape_function_RT(fe.base_element(1),
    //                                                           global_cell_it,
    //                                                           /*verbose
    //                                                           =*/false);
    ShapeFun::BasisRaviartThomas<3> std_shape_function_RT(global_cell_it,
                                                          /* degree */ 0);

    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);

    for (unsigned int n_basis = 0; n_basis < basis_div_v.size(); ++n_basis)
      {
        //        std_shape_function_RT.set_shape_fun_index(n_basis);
        std_shape_function_RT.set_index(n_basis);

        // set constraints (first hanging nodes, then flux)
        constraints_div_v[n_basis].clear();

        DoFTools::make_hanging_node_constraints(dof_handler,
                                                constraints_div_v[n_basis]);

        VectorTools::project_boundary_values_curl_conforming_l2(
          dof_handler,
          /*first vector component */ 0,
          Functions::ZeroFunction<3>(6), // This is not so important as long as
                                         // BCs do not influence u.
          /*boundary id*/ 0,
          constraints_div_v[n_basis]);
        VectorTools::project_boundary_values_div_conforming(
          dof_handler,
          /*first vector component */ 3,
          std_shape_function_RT, // This is important
          /*boundary id*/ 0,
          constraints_div_v[n_basis]);

        constraints_div_v[n_basis].close();
      }

    for (unsigned int n_basis = 0; n_basis < basis_div_v.size(); ++n_basis)
      {
        basis_div_v[n_basis].reinit(dofs_per_block);
        system_rhs_div_v[n_basis].reinit(dofs_per_block);
      }

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

  void
    NedRTBasis::assemble_system()
  {
    Timer timer;
    if (parameters.verbose)
      {
        std::cout << "	Assembling local linear system in cell   "
                  << global_cell_id.to_string() << ".....";

        timer.restart();
      }
    // Choose appropriate quadrature rules
    QGauss<3> quadrature_formula(parameters.degree + 2);

    // Get relevant quantities to be updated from finite element
    FEValues<3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    // Define some abbreviations
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    // Declare local contributions and reserve memory
    FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>              local_rhs(dofs_per_cell);
    std::vector<Vector<double>> local_rhs_v(GeometryInfo<3>::lines_per_cell +
                                              GeometryInfo<3>::faces_per_cell,
                                            Vector<double>(dofs_per_cell));

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // equation data
    std::unique_ptr<EquationData::RightHandSide> right_hand_side;
    if (parameters.use_exact_solution)
      right_hand_side.reset(
        new EquationData::RightHandSideExactLin(parameter_filename));
    else
      right_hand_side.reset(
        new EquationData::RightHandSideParsed(parameter_filename,
                                              /* n_components */ 3));

    const EquationData::DiffusionInverse_A diffusion_inverse_a(
      parameter_filename);
    const EquationData::Diffusion_B  diffusion_b(parameter_filename,
                                                parameters.use_exact_solution);
    const EquationData::ReactionRate reaction_rate;

    // allocate
    std::vector<Tensor<1, 3>> rhs_values(n_q_points);
    std::vector<Tensor<2, 3>> diffusion_inverse_a_values(n_q_points);
    std::vector<double>       diffusion_b_values(n_q_points);
    std::vector<double>       reaction_rate_values(n_q_points);

    ////////////////////////////////////////
    ShapeFun::BasisNedelecCurl<3> std_shape_function_Ned_curl(global_cell_it,
                                                              /* degree */ 0);
    std::vector<std::vector<Tensor<1, 3>>> local_rhs_values_curl(
      GeometryInfo<3>::lines_per_cell, std::vector<Tensor<1, 3>>(n_q_points));
    ////////////////////////////////////////

    const FEValuesExtractors::Vector curl(/* first_vector_component */ 0);
    const FEValuesExtractors::Vector flux(/* first_vector_component */ 3);

    // ------------------------------------------------------------------
    // loop over cells
    typename DoFHandler<3>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        for (unsigned int n_basis = 0; n_basis < length_system_basis; ++n_basis)
          {
            local_rhs_v[n_basis] = 0;

            if ((n_basis < GeometryInfo<3>::lines_per_cell) &&
                (parameters.full_rhs))
              {
                std_shape_function_Ned_curl.set_index(n_basis);

                std_shape_function_Ned_curl.tensor_value_list(
                  fe_values.get_quadrature_points(),
                  local_rhs_values_curl[n_basis]);
              }
          }

        right_hand_side->tensor_value_list(fe_values.get_quadrature_points(),
                                           rhs_values);
        reaction_rate.value_list(fe_values.get_quadrature_points(),
                                 reaction_rate_values);
        diffusion_inverse_a.value_list(fe_values.get_quadrature_points(),
                                       diffusion_inverse_a_values);
        diffusion_b.value_list(fe_values.get_quadrature_points(),
                               diffusion_b_values);

        // loop over quad points
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // Test functions
                const Tensor<1, 3> tau_i      = fe_values[curl].value(i, q);
                const Tensor<1, 3> curl_tau_i = fe_values[curl].curl(i, q);
                const double       div_v_i = fe_values[flux].divergence(i, q);
                const Tensor<1, 3> v_i     = fe_values[flux].value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // trial functions
                    const Tensor<1, 3> sigma_j = fe_values[curl].value(j, q);
                    const Tensor<1, 3> curl_sigma_j =
                      fe_values[curl].curl(j, q);
                    const double div_u_j   = fe_values[flux].divergence(j, q);
                    const Tensor<1, 3> u_j = fe_values[flux].value(j, q);

                    /*
                     * Discretize
                     * A^{-1}sigma - curl(u) = 0
                     * curl(sigma) - grad(B*div(u)) + alpha
                     * u = f , where alpha>0.
                     */
                    local_matrix(i, j) +=
                      (tau_i * diffusion_inverse_a_values[q] *
                         sigma_j            /* block (0,0) */
                       - curl_tau_i * u_j   /* block (0,1) */
                       + v_i * curl_sigma_j /* block (1,0) */
                       + div_v_i * diffusion_b_values[q] *
                           div_u_j                            /* block (1,1) */
                       + v_i * reaction_rate_values[q] * u_j) /* block (1,1) */
                      * fe_values.JxW(q);
                  } // end for ++j

                // Only for use in global assembly
                local_rhs(i) += v_i * rhs_values[q] * fe_values.JxW(q);

                // Only for use in local solving.
                if (parameters.full_rhs)
                  for (unsigned int n_basis = 0;
                       n_basis < GeometryInfo<3>::lines_per_cell;
                       ++n_basis)
                    {
                      local_rhs_v[n_basis](i) +=
                        v_i * local_rhs_values_curl[n_basis][q] *
                        fe_values.JxW(q);
                    }
              } // end for ++i
          }     // end for ++q

        //		for (unsigned int face_number=0;
        //				face_number<GeometryInfo<3>::faces_per_cell;
        //				++face_number)
        //		{
        //			if (cell->face(face_number)->at_boundary()
        ////				&&
        ////				(cell->face(face_number)->boundary_id() ==
        /// 0)
        //				)
        //			{
        //				fe_face_values.reinit (cell,
        // face_number);
        //		.....
        //			} // end if cell->at_boundary()
        //		} // end face_number++

        // Only for use in global assembly
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            global_rhs(local_dof_indices[i]) += local_rhs(i);
          }

        // Add to global matrix
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                assembled_matrix.add(local_dof_indices[i],
                                     local_dof_indices[j],
                                     local_matrix(i, j));
              }

            for (unsigned int n_basis = 0; n_basis < length_system_basis;
                 ++n_basis)
              {
                if (n_basis < GeometryInfo<3>::lines_per_cell)
                  {
                    // This is for curl.
                    system_rhs_curl_v[n_basis](local_dof_indices[i]) +=
                      local_rhs_v[n_basis](i);
                  }
                else
                  {
                    // This is for curl.
                    const unsigned int offset_index =
                      n_basis - GeometryInfo<3>::lines_per_cell;
                    system_rhs_div_v[offset_index](local_dof_indices[i]) +=
                      local_rhs_v[n_basis](i);
                  }
              }
          }
        // ------------------------------------------
      } // end for ++cell

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  } // end assemble()

  void
    NedRTBasis::solve_direct(unsigned int n_basis)
  {
    Timer timer;
    if (parameters.verbose)
      {
        std::cout << "	Solving linear system (SparseDirectUMFPACK) in cell   "
                  << global_cell_id.to_string() << "   for basis   " << n_basis
                  << ".....";

        timer.restart();
      }

    BlockVector<double> *system_rhs_ptr = NULL;
    BlockVector<double> *solution_ptr   = NULL;
    if (n_basis < GeometryInfo<3>::lines_per_cell)
      {
        system_rhs_ptr = &(system_rhs_curl_v[n_basis]);
        solution_ptr   = &(basis_curl_v[n_basis]);
      }
    else
      {
        const unsigned int offset_index =
          n_basis - GeometryInfo<3>::lines_per_cell;
        system_rhs_ptr = &(system_rhs_div_v[offset_index]);
        solution_ptr   = &(basis_div_v[offset_index]);
      }

    // for convenience
    const BlockVector<double> &system_rhs = *system_rhs_ptr;
    BlockVector<double> &      solution   = *solution_ptr;

    // use direct solver
    SparseDirectUMFPACK A_inv;
    A_inv.initialize(system_matrix);

    A_inv.vmult(solution, system_rhs);

    if (n_basis < GeometryInfo<3>::lines_per_cell)
      {
        constraints_curl_v[n_basis].distribute(solution);
      }
    else
      {
        const unsigned int offset_index =
          n_basis - GeometryInfo<3>::lines_per_cell;

        constraints_div_v[offset_index].distribute(solution);
      }

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

  void
    NedRTBasis::solve_iterative(unsigned int n_basis)
  {
    Timer timer;
    Timer inner_timer;

    // ------------------------------------------
    // Make a preconditioner for each system matrix
    if (parameters.verbose)
      {
        std::cout << "	Computing preconditioner in cell   "
                  << global_cell_id.to_string() << "   for basis   " << n_basis
                  << "   .....";

        timer.restart();
      }

    BlockVector<double> *system_rhs_ptr = NULL;
    BlockVector<double> *solution_ptr   = NULL;
    if (n_basis < GeometryInfo<3>::lines_per_cell)
      {
        system_rhs_ptr = &(system_rhs_curl_v[n_basis]);
        solution_ptr   = &(basis_curl_v[n_basis]);
      }
    else
      {
        const unsigned int offset_index =
          n_basis - GeometryInfo<3>::lines_per_cell;
        system_rhs_ptr = &(system_rhs_div_v[offset_index]);
        solution_ptr   = &(basis_div_v[offset_index]);
      }

    // for convenience
    const BlockVector<double> &system_rhs = *system_rhs_ptr;
    BlockVector<double> &      solution   = *solution_ptr;

    inner_schur_preconditioner = std::make_shared<
      typename LinearSolvers::LocalInnerPreconditioner<3>::type>();

    typename LinearSolvers::LocalInnerPreconditioner<3>::type::AdditionalData
      data;
    inner_schur_preconditioner->initialize(system_matrix.block(0, 0), data);

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
    // ------------------------------------------

    // Now solve.
    if (parameters.verbose)
      {
        std::cout << "	Solving linear system (iteratively, with "
                     "preconditioner) in cell   "
                  << global_cell_id.to_string() << "   for basis   " << n_basis
                  << "   .....";

        timer.restart();
      }

    // Construct inverse of upper left block
    const LinearSolvers::InverseMatrix<
      SparseMatrix<double>,
      typename LinearSolvers::LocalInnerPreconditioner<3>::type>
      block_inverse(system_matrix.block(0, 0), *inner_schur_preconditioner);

    Vector<double> tmp(system_rhs.block(0).size());
    {
      // Set up Schur complement
      LinearSolvers::SchurComplement<
        BlockSparseMatrix<double>,
        Vector<double>,
        typename LinearSolvers::LocalInnerPreconditioner<3>::type>
        schur_complement(system_matrix, block_inverse);

      // Compute schur_rhs = -g + C*A^{-1}*f
      Vector<double> schur_rhs(system_rhs.block(1).size());

      block_inverse.vmult(tmp, system_rhs.block(0));
      system_matrix.block(1, 0).vmult(schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);

      {
        if (parameters.verbose)
          {
            inner_timer.restart();
          }
        SolverControl            solver_control(system_matrix.m(),
                                     1e-6 * schur_rhs.l2_norm());
        SolverCG<Vector<double>> schur_solver(solver_control);

        //			PreconditionIdentity preconditioner;

        /*
         * Precondition the Schur complement with
         * the approximate inverse of the
         * Schur complement.
         */
        //			LinearSolvers::ApproximateInverseMatrix<LinearSolvers::SchurComplement<BlockSparseMatrix<double>,
        //																					Vector<double>,
        //																					typename
        // LinearSolvers::LocalInnerPreconditioner<3>::type>,
        //										PreconditionIdentity>
        //										preconditioner
        //(schur_complement, PreconditionIdentity(),
        //													/* n_iter */
        // 5);

        /*
         * Precondition the Schur complement with
         * the (approximate) inverse of an approximate
         * Schur complement.
         */
        //			using ApproxSchurPrecon =
        // PreconditionJacobi<SparseMatrix<double>>;
        // using ApproxSchurPrecon =
        // PreconditionSOR<SparseMatrix<double>>;
        using ApproxSchurPrecon = SparseILU<double>;
        //			using ApproxSchurPrecon = PreconditionIdentity;
        LinearSolvers::ApproximateSchurComplement<BlockSparseMatrix<double>,
                                                  Vector<double>,
                                                  ApproxSchurPrecon>
          approx_schur(system_matrix);

        LinearSolvers::ApproximateInverseMatrix<
          LinearSolvers::ApproximateSchurComplement<BlockSparseMatrix<double>,
                                                    Vector<double>,
                                                    ApproxSchurPrecon>,
          PreconditionIdentity>
          preconditioner(approx_schur,
                         PreconditionIdentity(),
#ifdef DEBUG
                         /* n_iter */ 1000);
#else
                         /* n_iter */ 14);
#endif

        schur_solver.solve(schur_complement,
                           solution.block(1),
                           schur_rhs,
                           preconditioner);

        if (n_basis < GeometryInfo<3>::lines_per_cell)
          {
            constraints_curl_v[n_basis].distribute(solution);
          }
        else
          {
            const unsigned int offset_index =
              n_basis - GeometryInfo<3>::lines_per_cell;

            constraints_div_v[offset_index].distribute(solution);
          }

        if (parameters.verbose)
          {
            inner_timer.stop();

            std::cout << std::endl
                      << "		- Iterative Schur complement solver "
                         "converged in   "
                      << solver_control.last_step()
                      << "   iterations.	Time:	" << inner_timer.cpu_time()
                      << "   seconds." << std::endl;
          }
      }

      {
        if (parameters.verbose)
          {
            inner_timer.restart();
          }

        // use computed u to solve for sigma
        system_matrix.block(0, 1).vmult(tmp, solution.block(1));
        tmp *= -1;
        tmp += system_rhs.block(0);

        // Solve for sigma
        block_inverse.vmult(solution.block(0), tmp);

        if (parameters.verbose)
          {
            inner_timer.stop();

            std::cout << "		- Outer solver completed.   Time:   "
                      << inner_timer.cpu_time() << "   seconds." << std::endl;
          }
      }

      if (n_basis < GeometryInfo<3>::lines_per_cell)
        {
          constraints_curl_v[n_basis].distribute(solution);
        }
      else
        {
          const unsigned int offset_index =
            n_basis - GeometryInfo<3>::lines_per_cell;

          constraints_div_v[offset_index].distribute(solution);
        }
    }

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "		- done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

  void
    NedRTBasis::assemble_global_element_matrix()
  {
    // First, reset.
    global_element_matrix = 0;

    // Get lengths of tmp vectors for assembly
    std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
    const unsigned int n_sigma = dofs_per_component[0],
                       n_u     = dofs_per_component[3];

    Vector<double> tmp_u(n_u), tmp_sigma(n_sigma);

    // This assembles the local contribution to the global global matrix
    // with an algebraic trick. It uses the local system matrix stored in
    // the respective basis object.
    unsigned int block_row, block_col;

    BlockVector<double> *test_vec_ptr, *trial_vec_ptr;
    unsigned int         offset_index = GeometryInfo<3>::lines_per_cell;

    for (unsigned int i_test = 0; i_test < length_system_basis; ++i_test)
      {
        if (i_test < GeometryInfo<3>::lines_per_cell)
          {
            block_row    = 0;
            test_vec_ptr = &(basis_curl_v.at(i_test));
          }
        else
          {
            block_row    = 1;
            test_vec_ptr = &(basis_div_v.at(i_test - offset_index));
          }

        for (unsigned int i_trial = 0; i_trial < length_system_basis; ++i_trial)
          {
            if (i_trial < GeometryInfo<3>::lines_per_cell)
              {
                block_col     = 0;
                trial_vec_ptr = &(basis_curl_v.at(i_trial));
              }
            else
              {
                block_col     = 1;
                trial_vec_ptr = &(basis_div_v.at(i_trial - offset_index));
              }

            if (block_row == 0) /* This means we are testing with sigma. */
              {
                if (block_col == 0) /* This means trial function is sigma. */
                  {
                    assembled_matrix.block(block_row, block_col)
                      .vmult(tmp_sigma, trial_vec_ptr->block(block_col));
                    global_element_matrix(i_test, i_trial) +=
                      (test_vec_ptr->block(block_row) * tmp_sigma);
                    tmp_sigma = 0;
                  }
                if (block_col == 1) /* This means trial function is u. */
                  {
                    assembled_matrix.block(block_row, block_col)
                      .vmult(tmp_sigma, trial_vec_ptr->block(block_col));
                    global_element_matrix(i_test, i_trial) +=
                      (test_vec_ptr->block(block_row) * tmp_sigma);
                    tmp_sigma = 0;
                  }
              }  // end if
            else /* This means we are testing with u. */
              {
                if (block_col == 0) /* This means trial function is sigma. */
                  {
                    assembled_matrix.block(block_row, block_col)
                      .vmult(tmp_u, trial_vec_ptr->block(block_col));
                    global_element_matrix(i_test, i_trial) +=
                      (test_vec_ptr->block(block_row) * tmp_u);
                    tmp_u = 0;
                  }
                if (block_col == 1) /* This means trial function is u. */
                  {
                    assembled_matrix.block(block_row, block_col)
                      .vmult(tmp_u, trial_vec_ptr->block(block_col));
                    global_element_matrix(i_test, i_trial) +=
                      test_vec_ptr->block(block_row) * tmp_u;
                    tmp_u = 0;
                  }
              } // end else
          }     // end for i_trial

        if (i_test >= GeometryInfo<3>::lines_per_cell)
          {
            block_row = 1;
            // If we are testing with u we possibly have a
            // right-hand side.
            global_element_rhs(i_test) +=
              test_vec_ptr->block(block_row) * global_rhs.block(block_row);
          }
      } // end for i_test

    is_built_global_element_matrix = true;
  }

  void
    NedRTBasis::output_basis()
  {
    Timer timer;
    if (parameters.verbose)
      {
        std::cout << "	Writing local basis in cell   "
                  << global_cell_id.to_string() << ".....";

        timer.restart();
      }

    for (unsigned int n_basis = 0; n_basis < length_system_basis; ++n_basis)
      {
        BlockVector<double> *basis_ptr = NULL;
        if (n_basis < GeometryInfo<3>::lines_per_cell)
          basis_ptr = &(basis_curl_v.at(n_basis));
        else
          basis_ptr =
            &(basis_div_v.at(n_basis - GeometryInfo<3>::lines_per_cell));

        std::vector<std::string> solution_names(3, "sigma");
        solution_names.push_back("u");
        solution_names.push_back("u");
        solution_names.push_back("u");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          interpretation(
            3, DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(
          DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(
          DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(
          DataComponentInterpretation::component_is_part_of_vector);

        NedRT_PostProcessor postprocessor(parameter_filename,
                                          parameters.use_exact_solution);

        DataOut<3> data_out;
        data_out.attach_dof_handler(dof_handler);

        data_out.add_data_vector(*basis_ptr,
                                 solution_names,
                                 DataOut<3>::type_dof_data,
                                 interpretation);
        data_out.add_data_vector(*basis_ptr, postprocessor);

        data_out.build_patches(parameters.degree + 1);

        // filename
        std::string filename = "basis_ned-rt";
        if (n_basis < GeometryInfo<3>::lines_per_cell)
          {
            filename += ".curl";
            filename += "." + Utilities::int_to_string(local_subdomain, 5);
            filename += ".cell-" + global_cell_id.to_string();
            filename += ".index-";
            filename += Utilities::int_to_string(n_basis, 2);
          }
        else
          {
            filename += ".div";
            filename += "." + Utilities::int_to_string(local_subdomain, 5);
            filename += ".cell-" + global_cell_id.to_string();
            filename += ".index-";
            filename += Utilities::int_to_string(
              n_basis - GeometryInfo<3>::lines_per_cell, 2);
          }
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
    NedRTBasis::write_exact_solution_in_cell()
  {
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);
    exact_solution_in_cell.reinit(dofs_per_block);

    { // write sigma
      // Quadrature used for projection
      QGauss<3> quad_rule(3);

      // Setup function
      EquationData::ExactSolutionLin_A_curl exact_sigma(parameter_filename);

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
                                         exact_solution_in_cell.block(0));

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
                                         exact_solution_in_cell.block(1));

      dof_handler_fake.clear();
    }
  }

  void
    NedRTBasis::output_global_solution_in_cell()
  {
    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names(3, "sigma");
    solution_names.emplace_back("u");
    solution_names.emplace_back("u");
    solution_names.emplace_back("u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        3 + 3, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(global_solution,
                             solution_names,
                             DataOut<3>::type_dof_data,
                             data_component_interpretation);

    // Postprocess
    std::unique_ptr<NedRT_PostProcessor> postprocessor(
      new NedRT_PostProcessor(parameter_filename,
                              parameters.use_exact_solution));
    data_out.add_data_vector(global_solution, *postprocessor);

    // Postprocess exact solution if needed
    std::vector<std::string> solution_names_exact(3, "exact_sigma");
    solution_names_exact.emplace_back("exact_u");
    solution_names_exact.emplace_back("exact_u");
    solution_names_exact.emplace_back("exact_u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_exact(
        3 + 3, DataComponentInterpretation::component_is_part_of_vector);

    if (parameters.use_exact_solution)
      {
        data_out.add_data_vector(exact_solution_in_cell,
                                 solution_names_exact,
                                 DataOut<3>::type_dof_data,
                                 data_component_interpretation_exact);
      }

    std::unique_ptr<NedRT_PostProcessor> postprocessor_exact;
    if (parameters.use_exact_solution)
      {
        postprocessor_exact.reset(new NedRT_PostProcessor(
          parameter_filename, parameters.use_exact_solution, "exact_"));
        data_out.add_data_vector(exact_solution_in_cell, *postprocessor_exact);
      } // end if (parameters.use_exact_solution)

    data_out.build_patches();

    std::ofstream output(parameters.dirname_output + "/" +
                         parameters.filename_global);
    data_out.write_vtu(output);
  }

  void
    NedRTBasis::set_output_flag()
  {
    parameters.set_output_flag(global_cell_id, first_cell);
  }

  void
    NedRTBasis::set_global_weights(const std::vector<double> &weights)
  {
    // Copy assignment of global weights
    global_weights = weights;

    // reinitialize the global solution on this cell
    global_solution = 0;

    const unsigned int dofs_per_cell_sigma =
      fe.base_element(0).n_dofs_per_cell();
    const unsigned int dofs_per_cell_u = fe.base_element(1).n_dofs_per_cell();

    // First set block 0
    for (unsigned int i = 0; i < dofs_per_cell_sigma; ++i)
      global_solution.block(0).sadd(1,
                                    global_weights[i],
                                    basis_curl_v[i].block(0));

    // Then set block 1
    for (unsigned int i = 0; i < dofs_per_cell_u; ++i)
      global_solution.block(1).sadd(1,
                                    global_weights[i + dofs_per_cell_sigma],
                                    basis_div_v[i].block(1));

    is_set_global_weights = true;
  }

  void
    NedRTBasis::set_sigma_to_std()
  {
    // Quadrature used for projection
    QGauss<3> quad_rule(3);

    // Set up vector shape function from finite element on current cell
    ShapeFun::BasisNedelec<3> std_shape_function_curl(global_cell_it,
                                                      /* degree */ 0);

    DoFHandler<3> dof_handler_fake(triangulation);
    dof_handler_fake.distribute_dofs(fe.base_element(0));

    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_fake, constraints);
    constraints.close();

    for (unsigned int i = 0; i < basis_curl_v.size(); ++i)
      {
        basis_curl_v[i].block(0).reinit(dof_handler_fake.n_dofs());
        basis_curl_v[i].block(1) = 0;

        std_shape_function_curl.set_index(i);

        VectorTools::project(dof_handler_fake,
                             constraints,
                             quad_rule,
                             std_shape_function_curl,
                             basis_curl_v[i].block(0));
      }

    dof_handler_fake.clear();
  }

  void
    NedRTBasis::set_u_to_std()
  {
    // Quadrature used for projection
    QGauss<3> quad_rule(3);

    // Set up vector shape function from finite element on current cell
    ShapeFun::BasisRaviartThomas<3> std_shape_function_div(global_cell_it,
                                                           /*degree*/ 0);

    DoFHandler<3> dof_handler_fake(triangulation);
    dof_handler_fake.distribute_dofs(fe.base_element(1));

    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_fake, constraints);
    constraints.close();

    for (unsigned int i = 0; i < basis_div_v.size(); ++i)
      {
        basis_div_v[i].block(0) = 0;
        basis_div_v[i].block(1).reinit(dof_handler_fake.n_dofs());

        std_shape_function_div.set_index(i);

        VectorTools::project(dof_handler_fake,
                             constraints,
                             quad_rule,
                             std_shape_function_div,
                             basis_div_v[i].block(1));
      }

    dof_handler_fake.clear();
  }

  void
    NedRTBasis::set_filename_global()
  {
    parameters.filename_global +=
      ("." + Utilities::int_to_string(local_subdomain, 5) + ".cell-" +
       global_cell_id.to_string() + ".vtu");
  }

  const FullMatrix<double> &
    NedRTBasis::get_global_element_matrix() const
  {
    return global_element_matrix;
  }

  const Vector<double> &
    NedRTBasis::get_global_element_rhs() const
  {
    return global_element_rhs;
  }

  const std::string &
    NedRTBasis::get_filename_global() const
  {
    return parameters.filename_global;
  }

  void
    NedRTBasis::run()
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

    // Create grid
    setup_grid();

    // Reserve space for system matrices
    setup_system_matrix();

    // Set up boundary conditions and other constraints
    setup_basis_dofs_curl();
    setup_basis_dofs_div();

    // Assemble
    assemble_system();

    if (parameters.set_to_std)
      {
        Assert(parameters.renumber_dofs == false,
               ExcMessage("Dof renumbering messes up the basis."));

        Timer timer;
        if (parameters.verbose)
          {
            std::cout << "      Setting basis functions to standard functions. "
                         "This is slow"
                      << ".....";

            timer.restart();
          }
        set_sigma_to_std(); /* This is only a sanity check. */
        set_u_to_std();     /* This is only a sanity check. */

        if (parameters.verbose)
          {
            timer.stop();
            std::cout << "done in   " << timer.cpu_time() << "   seconds."
                      << std::endl;
          }
      }
    else // in this case solve
      {
        for (unsigned int n_basis = 0; n_basis < length_system_basis; ++n_basis)
          {
            if (n_basis < GeometryInfo<3>::lines_per_cell)
              {
                // This is for curl.
                system_matrix.reinit(sparsity_pattern);

                system_matrix.copy_from(assembled_matrix);

                // Now take care of constraints
                constraints_curl_v[n_basis].condense(
                  system_matrix, system_rhs_curl_v[n_basis]);

                // Now solve
                if (parameters.use_direct_solver)
                  solve_direct(n_basis);
                else
                  {
                    solve_iterative(n_basis);
                  }
              }
            else
              {
                // This is for div.
                const unsigned int offset_index =
                  n_basis - GeometryInfo<3>::lines_per_cell;

                system_matrix.reinit(sparsity_pattern);

                system_matrix.copy_from(assembled_matrix);

                // Now take care of constraints
                constraints_div_v[offset_index].condense(
                  system_matrix, system_rhs_div_v[offset_index]);

                // Now solve
                if (parameters.use_direct_solver)
                  solve_direct(n_basis);
                else
                  {
                    solve_iterative(n_basis);
                  }
              }
          }
      }

    if (parameters.use_exact_solution)
      write_exact_solution_in_cell();

    assemble_global_element_matrix();

    {
      // Free memory as much as possible
      system_matrix.clear();
      sparsity_pattern.reinit(0, 0);
      for (unsigned int i = 0; i < basis_curl_v.size(); ++i)
        {
          constraints_curl_v[i].clear();
        }
      for (unsigned int i = 0; i < basis_div_v.size(); ++i)
        {
          constraints_div_v[i].clear();
        }
    }

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

    if (true)
      {
        timer.stop();

        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

} // end namespace NedRT
