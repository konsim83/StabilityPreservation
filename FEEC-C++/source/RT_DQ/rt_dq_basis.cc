#include <RT_DQ/rt_dq_basis.h>

namespace RTDQ
{
  using namespace dealii;

  RTDQBasis::RTDQBasis(
    const ParametersMs &                             parameters_ms,
    const std::string &                              parameter_filename_,
    typename Triangulation<3>::active_cell_iterator &global_cell,
    CellId                                           first_cell,
    unsigned int                                     local_subdomain,
    MPI_Comm                                         mpi_communicator)
    : mpi_communicator(mpi_communicator)
    , parameters(parameters_ms)
    , parameter_filename(parameter_filename_)
    , triangulation()
    , fe(FE_RaviartThomas<3>(parameters.degree),
         1,
         FE_DGQ<3>(parameters.degree),
         1)
    , dof_handler(triangulation)
    , constraints_div_v(GeometryInfo<3>::faces_per_cell)
    , sparsity_pattern()
    , basis_div_v(GeometryInfo<3>::faces_per_cell)
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
    , length_system_basis(GeometryInfo<3>::faces_per_cell + 1)
    , is_built_global_element_matrix(false)
    , is_set_global_weights(false)
    , is_set_cell_data(false)
    , is_copyable(true)
  {
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
        //        face_measure.at(j_face) =
        //        global_cell_it->face(j_face)->measure();
      }

    for (unsigned int j_egde = 0; j_egde < GeometryInfo<3>::lines_per_cell;
         ++j_egde)
      {
        //        edge_measure.at(j_egde) =
        //        global_cell_it->line(j_egde)->measure();
      }

    is_set_cell_data = true;
  }

  RTDQBasis::RTDQBasis(const RTDQBasis &other)
    : mpi_communicator(other.mpi_communicator)
    , parameters(other.parameters)
    , parameter_filename(other.parameter_filename)
    , triangulation() // must be constructed deliberately, but is empty on
                      // copying anyway
    , fe(FE_RaviartThomas<3>(parameters.degree),
         1,
         FE_DGQ<3>(parameters.degree),
         1)
    , dof_handler(triangulation)
    , constraints_div_v(other.constraints_div_v)
    //    , sparsity_pattern(
    //        other.sparsity_pattern) // only possible if object is empty
    //    , assembled_matrix(
    //        other.assembled_matrix)          // only possible if object is
    //        empty
    //    , system_matrix(other.system_matrix) // only possible if object is
    //    empty
    , basis_div_v(other.basis_div_v)
    , system_rhs_div_v(other.system_rhs_div_v)
    , global_rhs(other.global_rhs)
    , global_element_matrix(other.global_element_matrix)
    , global_element_rhs(other.global_element_rhs)
    , global_weights(other.global_weights)
    , global_solution(other.global_solution)
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
        //        face_measure.at(j_face) =
        //        global_cell_it->face(j_face)->measure();
      }

    for (unsigned int j_egde = 0; j_egde < GeometryInfo<3>::lines_per_cell;
         ++j_egde)
      {
        //        edge_measure.at(j_egde) =
        //        global_cell_it->line(j_egde)->measure();
      }

    is_set_cell_data = true;
  }

  RTDQBasis::~RTDQBasis()
  {
    system_matrix.clear();

    for (unsigned int n_basis = 0; n_basis < GeometryInfo<3>::faces_per_cell;
         ++n_basis)
      {
        constraints_div_v[n_basis].clear();
      }

    dof_handler.clear();
  }

  void
    RTDQBasis::setup_grid()
  {
    Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

    GridGenerator::general_cell(
      triangulation,
      corner_points,
      /* colorize faces */ parameters.fast_constraint_setup);

    if (parameters.fast_constraint_setup)
      {
        typename Triangulation<3>::cell_iterator cell = triangulation.begin(),
                                                 endc = triangulation.end();
        for (; cell != endc; ++cell)
          {
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<3>::faces_per_cell;
                 face_number++)
              {
                if (std::fabs(cell->face(face_number)->center()(0) -
                              corner_points.at(0)(0)) < 1e-12)
                  {
                    // x=0
                    cell->face(face_number)->set_boundary_id(0);
                  } // end if

                else if (std::fabs(cell->face(face_number)->center()(0) -
                                   corner_points.at(1)(0)) < 1e-12)
                  {
                    // x=1
                    cell->face(face_number)->set_boundary_id(1);
                  }

                else if (std::fabs(cell->face(face_number)->center()(1) -
                                   corner_points.at(0)(1)) < 1e-12)
                  {
                    // y=0
                    cell->face(face_number)->set_boundary_id(2);
                  }

                else if (std::fabs(cell->face(face_number)->center()(1) -
                                   corner_points.at(2)(1)) < 1e-12)
                  {
                    // y=1
                    cell->face(face_number)->set_boundary_id(3);
                  }

                else if (std::fabs(cell->face(face_number)->center()(2) -
                                   corner_points.at(0)(2)) < 1e-12)
                  {
                    // z=0
                    cell->face(face_number)->set_boundary_id(4);
                  }

                else if (std::fabs(cell->face(face_number)->center()(2) -
                                   corner_points.at(4)(2)) < 1e-12)
                  {
                    // z=1
                    cell->face(face_number)->set_boundary_id(5);
                  }
              } // end for ++face_number
          }     // end for ++cell
      }

    triangulation.refine_global(parameters.n_refine_local);

    is_copyable = false;
  }

  void
    RTDQBasis::setup_system_matrix()
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
    RTDQBasis::setup_basis_dofs_div()
  {
    Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

    Timer timer;

    if (parameters.verbose)
      {
        std::cout << "	Setting up dofs for H(div) part.....";

        timer.restart();
      }

    ShapeFun::BasisRaviartThomas<3> std_shape_function_RT(global_cell_it,
                                                          /* degree */ 0);

    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);

    for (unsigned int n_basis = 0; n_basis < GeometryInfo<3>::faces_per_cell;
         ++n_basis)
      {
        // set constraints (first hanging nodes, then flux)
        constraints_div_v[n_basis].clear();

        DoFTools::make_hanging_node_constraints(dof_handler,
                                                constraints_div_v[n_basis]);

        if (parameters.fast_constraint_setup)
          {
            std::cerr
              << "Fast constraint setup discarded. This does not work on curved domains."
              << std::endl;
            exit(1);
            //            // boundary values for normal flux
            //            const unsigned int dofs_per_cell = fe.dofs_per_cell;
            //            const unsigned int dofs_per_face = fe.dofs_per_face;
            //            std::vector<types::global_dof_index>
            //            local_dof_indices(
            //              dofs_per_cell);
            //            std::vector<types::global_dof_index>
            //            local_dof_face_indices(
            //              dofs_per_face);
            //
            //            typename DoFHandler<3>::active_cell_iterator
            //              cell = dof_handler.begin_active(),
            //              endc = dof_handler.end();
            //            for (; cell != endc; ++cell)
            //              {
            //                if (cell->at_boundary())
            //                  {
            //                    cell->get_dof_indices(local_dof_indices);
            //                    for (unsigned int face_n = 0;
            //                         face_n < GeometryInfo<3>::faces_per_cell;
            //                         ++face_n)
            //                      {
            //                        if (cell->at_boundary(face_n))
            //                          {
            //                            cell->face(face_n)->get_dof_indices(
            //                              local_dof_face_indices);
            //                            if (cell->face(face_n)->boundary_id()
            //                            == n_basis)
            //                              {
            //                                for (unsigned int i = 0; i <
            //                                dofs_per_face; ++i)
            //                                  {
            //                                    const double dof_scale =
            //                                      cell->face(face_n)->measure()
            //                                      / face_measure.at(n_basis);
            //
            //                                    constraints_div_v[n_basis].add_line(
            //                                      local_dof_face_indices.at(i));
            //                                    constraints_div_v[n_basis]
            //                                      .set_inhomogeneity(
            //                                        local_dof_face_indices.at(i),
            //                                        dof_scale);
            //                                  }
            //                              }
            //                            else
            //                              {
            //                                for (unsigned int i = 0; i <
            //                                dofs_per_face; ++i)
            //                                  {
            //                                    constraints_div_v[n_basis].add_line(
            //                                      local_dof_face_indices.at(i));
            //                                  }
            //                              }
            //                          } // end
            //                            // cell->at_boundary(face_n)
            //                      }     // end ++face_n
            //                  }         // end if cell->at_boundary (
            //              }             // end ++cell
          }
        else
          {
            std_shape_function_RT.set_index(n_basis);

            VectorTools::project_boundary_values_div_conforming(
              dof_handler,
              /*first vector component */ 0,
              std_shape_function_RT, // This is important
              /*boundary id*/ 0,
              constraints_div_v[n_basis]);
          }

        constraints_div_v[n_basis].close();
      }

    for (unsigned int n_basis = 0; n_basis < GeometryInfo<3>::faces_per_cell;
         ++n_basis)
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
    RTDQBasis::assemble_system()
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
    std::vector<Vector<double>> local_rhs_v(GeometryInfo<3>::faces_per_cell,
                                            Vector<double>(dofs_per_cell));

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // equation data
    const EquationData::RightHandSideParsed right_hand_side(
      parameter_filename, /* n_components */ 1);
    const EquationData::DiffusionInverse_A a_inverse(parameter_filename);
    const EquationData::ReactionRate       reaction_rate;

    // allocate
    std::vector<double>       rhs_values(n_q_points);
    std::vector<double>       reaction_rate_values(n_q_points);
    std::vector<Tensor<2, 3>> a_inverse_values(n_q_points);

    // define extractors
    const FEValuesExtractors::Vector flux(0);
    const FEValuesExtractors::Scalar concentration(3);

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

        for (unsigned int n_basis = 0;
             n_basis < GeometryInfo<3>::faces_per_cell;
             ++n_basis)
          {
            local_rhs_v[n_basis] = 0;
          }

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);
        reaction_rate.value_list(fe_values.get_quadrature_points(),
                                 reaction_rate_values);
        a_inverse.value_list(fe_values.get_quadrature_points(),
                             a_inverse_values);

        // loop over quad points
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // Test functions
                const Tensor<1, 3> phi_i_sigma = fe_values[flux].value(i, q);
                const double div_phi_i_sigma = fe_values[flux].divergence(i, q);
                const double phi_i_u = fe_values[concentration].value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // Trial functions
                    const Tensor<1, 3> phi_j_sigma =
                      fe_values[flux].value(j, q);
                    const double div_phi_j_sigma =
                      fe_values[flux].divergence(j, q);
                    const double phi_j_u = fe_values[concentration].value(j, q);

                    /*
                     * Discretize
                     * K^{-1}sigma + grad(u) = 0
                     * div(sigma) + alpha*u = f , where
                     * alpha<0 (this is important) This is
                     * the simplest form of a
                     * diffusion-reaction equation where an
                     * anisotropic diffusion and reaction
                     * are in balance in a heterogeneous
                     * medium. A multiscale reaction rate is
                     * also possible and can easily be
                     * added.
                     */
                    local_matrix(i, j) +=
                      (phi_i_sigma * a_inverse_values[q] *
                         phi_j_sigma               /* Block (0, 0)*/
                       - div_phi_i_sigma * phi_j_u /* Block (0, 1)*/
                       + phi_i_u * div_phi_j_sigma /* Block (1, 0)*/
                       + reaction_rate_values[q] * phi_i_u *
                           phi_j_u /* Block (1, 1)*/
                       ) *
                      fe_values.JxW(q);
                  } // end for ++j

                // Only for use in global assembly
                local_rhs(i) += phi_i_u * rhs_values[q] * fe_values.JxW(q);

                // Only for use in local solving. Critical for
                // Darcy type problem. (Think of LBB between
                // RT0-DGQ0)
                for (unsigned int n_basis = 0;
                     n_basis < GeometryInfo<3>::faces_per_cell;
                     ++n_basis)
                  {
                    // Note the sign here.
                    if (parameters.is_laplace)
                      {
                        const double scale = 1 / volume_measure;
                        if (n_basis == 0)
                          local_rhs_v[n_basis](i) +=
                            -phi_i_u * scale * fe_values.JxW(q);
                        if (n_basis == 1)
                          local_rhs_v[n_basis](i) +=
                            phi_i_u * scale * fe_values.JxW(q);
                        if (n_basis == 2)
                          local_rhs_v[n_basis](i) +=
                            -phi_i_u * scale * fe_values.JxW(q);
                        if (n_basis == 3)
                          local_rhs_v[n_basis](i) +=
                            phi_i_u * scale * fe_values.JxW(q);
                        if (n_basis == 4)
                          local_rhs_v[n_basis](i) +=
                            -phi_i_u * scale * fe_values.JxW(q);
                        if (n_basis == 5)
                          local_rhs_v[n_basis](i) +=
                            phi_i_u * scale * fe_values.JxW(q);
                      }
                    else
                      local_rhs_v[n_basis](i) += 0;
                  }
              } // end for ++i
          }     // end for ++q

        // Only for use in global assembly.
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            global_rhs(local_dof_indices[i]) += local_rhs(i);
          }

        // Add to global matrix. Take care of constraints later.
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                assembled_matrix.add(local_dof_indices[i],
                                     local_dof_indices[j],
                                     local_matrix(i, j));
              }

            for (unsigned int n_basis = 0;
                 n_basis < GeometryInfo<3>::faces_per_cell;
                 ++n_basis)
              {
                system_rhs_div_v[n_basis](local_dof_indices[i]) +=
                  local_rhs_v[n_basis](i);
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
    RTDQBasis::solve_direct(unsigned int n_basis)
  {
    Timer timer;
    if (parameters.verbose)
      {
        std::cout << "	Solving linear system (SparseDirectUMFPACK) in cell   "
                  << global_cell_id.to_string() << "   for basis   " << n_basis
                  << ".....";

        timer.restart();
      }

    // for convenience define an alias
    const BlockVector<double> &system_rhs = system_rhs_div_v[n_basis];
    BlockVector<double> &      solution   = basis_div_v[n_basis];

    // use direct solver
    SparseDirectUMFPACK A_inv;
    A_inv.initialize(system_matrix);

    A_inv.vmult(solution, system_rhs);

    constraints_div_v[n_basis].distribute(solution);

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

  void
    RTDQBasis::solve_iterative(unsigned int n_basis)
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

    // for convenience define an alias
    const BlockVector<double> &system_rhs = system_rhs_div_v[n_basis];
    BlockVector<double> &      solution   = basis_div_v[n_basis];

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
        schur_complement(system_matrix, block_inverse, dof_handler);

      // Compute schur_rhs = -g + C*A^{-1}*f
      Vector<double> schur_rhs(system_rhs.block(1).size());

      block_inverse.vmult(tmp, system_rhs.block(0));
      system_matrix.block(1, 0).vmult(schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);

      if (parameters.is_laplace)
        {
          /*
           * Only used for Laplace problems since we then have
           * a pure Neumann problem and u is only determined up
           * to a constant.
           */
          DoFHandler<3>              fake_dof_u(this->triangulation);
          FEValuesExtractors::Scalar u_component(3);
          ComponentMask              u_mask(fe.component_mask(u_component));
          const auto &u_fe(dof_handler.get_fe().get_sub_fe(u_mask));
          fake_dof_u.distribute_dofs(u_fe);
          const double mean_value = VectorTools::compute_mean_value(
            fake_dof_u, QGauss<3>(2), schur_rhs, 0);
          schur_rhs.add(-mean_value);

          if (parameters.verbose)
            std::cout << std::endl
                      << "      Schur RHS pre-correction: The mean value"
                         "was adjusted by "
                      << -mean_value << "    -> new mean:   "
                      << VectorTools::compute_mean_value(fake_dof_u,
                                                         QGauss<3>(2),
                                                         schur_rhs,
                                                         0)
                      << std::endl;
        }


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

        constraints_div_v[n_basis].distribute(solution);

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

      constraints_div_v[n_basis].distribute(solution);
    }

    if (parameters.verbose)
      {
        timer.stop();
        std::cout << "		- done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  } // namespace RTDQ

  void
    RTDQBasis::assemble_global_element_matrix()
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

    for (unsigned int i_test = 0; i_test < length_system_basis; ++i_test)
      {
        test_vec_ptr =
          &(basis_div_v.at(i_test % GeometryInfo<3>::faces_per_cell));

        if (i_test < GeometryInfo<3>::faces_per_cell)
          block_row = 0;
        else
          block_row = 1;

        for (unsigned int i_trial = 0; i_trial < length_system_basis; ++i_trial)
          {
            trial_vec_ptr =
              &(basis_div_v.at(i_trial % GeometryInfo<3>::faces_per_cell));

            if (i_trial < GeometryInfo<3>::faces_per_cell)
              block_col = 0;
            else
              block_col = 1;

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

        if (i_test >= GeometryInfo<3>::faces_per_cell)
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
    RTDQBasis::output_basis()
  {
    Timer timer;
    if (parameters.verbose)
      {
        std::cout << "	Writing local basis in cell   "
                  << global_cell_id.to_string() << ".....";

        timer.restart();
      }

    for (unsigned int n_basis = 0; n_basis < GeometryInfo<3>::faces_per_cell;
         ++n_basis)
      {
        BlockVector<double> &basis = basis_div_v[n_basis];

        std::vector<std::string> solution_names(3, "sigma");
        solution_names.push_back("u");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          interpretation(
            3, DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(
          DataComponentInterpretation::component_is_scalar);

        DataOut<3> data_out;
        data_out.attach_dof_handler(dof_handler);

        data_out.add_data_vector(basis,
                                 solution_names,
                                 DataOut<3>::type_dof_data,
                                 interpretation);

        data_out.build_patches();

        // filename
        std::string filename = "basis_rt-dq";
        filename += ".div";
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
    RTDQBasis::output_global_solution_in_cell()
  {
    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names(3, "sigma");
    solution_names.emplace_back("u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        3, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(global_solution,
                             solution_names,
                             DataOut<3>::type_dof_data,
                             data_component_interpretation);

    // Postprocess
    std::unique_ptr<RTDQ_PostProcessor> postprocessor(
      new RTDQ_PostProcessor(parameter_filename));
    data_out.add_data_vector(global_solution, *postprocessor);

    data_out.build_patches();

    std::ofstream output(parameters.dirname_output + "/" +
                         parameters.filename_global);
    data_out.write_vtu(output);
  }

  void
    RTDQBasis::set_output_flag()
  {
    parameters.set_output_flag(global_cell_id, first_cell);
  }

  void
    RTDQBasis::set_global_weights(const std::vector<double> &weights)
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
                                    basis_div_v[i].block(0));

    // Then set block 1
    for (unsigned int i = 0; i < dofs_per_cell_u; ++i)
      global_solution.block(1).sadd(1,
                                    global_weights[i + dofs_per_cell_sigma],
                                    basis_div_v[i].block(1));

    is_set_global_weights = true;
  }

  void
    RTDQBasis::set_u_to_std()
  {
    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
      basis_div_v[i].block(1) = 1;
  }

  void
    RTDQBasis::set_sigma_to_std()
  {
    std::cout << "   (INFO: Sanity check for sigma employed!)   ";

    // Quadrature used for projection
    QGauss<3> quad_rule(3);

    // Set up vector shape function from finite element on current cell
    ShapeFun::BasisRaviartThomas<3> std_shape_function_div(global_cell_it,
                                                           /* degree */ 0);

    DoFHandler<3> dof_handler_fake(triangulation);
    dof_handler_fake.distribute_dofs(fe.base_element(0));

    if (parameters.renumber_dofs)
      {
        throw std::runtime_error("Renumbering DoFs not allowed when sanity "
                                 "checking basis for sigma.");
      }

    AffineConstraints<double> constraints;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_fake, constraints);
    constraints.close();

    for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
      {
        basis_div_v.at(i).block(0).reinit(dof_handler_fake.n_dofs());

        std_shape_function_div.set_index(i);

        VectorTools::project(dof_handler_fake,
                             constraints,
                             quad_rule,
                             std_shape_function_div,
                             basis_div_v[i].block(0));
      }

    dof_handler_fake.clear();
  }

  void
    RTDQBasis::set_filename_global()
  {
    parameters.filename_global +=
      ("." + Utilities::int_to_string(local_subdomain, 5) + ".cell-" +
       global_cell_id.to_string() + ".vtu");
  }

  const FullMatrix<double> &
    RTDQBasis::get_global_element_matrix() const
  {
    return global_element_matrix;
  }

  const Vector<double> &
    RTDQBasis::get_global_element_rhs() const
  {
    return global_element_rhs;
  }

  const std::string &
    RTDQBasis::get_filename_global() const
  {
    return parameters.filename_global;
  }

  void
    RTDQBasis::run()
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
    setup_basis_dofs_div();

    // Assemble
    assemble_system();

    for (unsigned int n_basis = 0; n_basis < GeometryInfo<3>::faces_per_cell;
         ++n_basis)
      {
        // This is for curl.
        system_matrix.reinit(sparsity_pattern);

        system_matrix.copy_from(assembled_matrix);

        // Now take care of constraints
        constraints_div_v[n_basis].condense(system_matrix,
                                            system_rhs_div_v[n_basis]);

        // Now solve
        if (parameters.use_direct_solver)
          solve_direct(n_basis);
        else
          {
            solve_iterative(n_basis);
          }
      }

    /*
     * This is necessary for stability. Must be
     * done before global assembly.
     */
    set_u_to_std();

    /*
     * This is only for sanity checking.
     */
    if (false)
      set_sigma_to_std();

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
      sparsity_pattern.reinit(0, 0);
      for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
        {
          constraints_div_v[i].clear();
        }
    }

    if (true)
      {
        timer.stop();

        std::cout << "done in   " << timer.cpu_time() << "   seconds."
                  << std::endl;
      }
  }

} // end namespace RTDQ
