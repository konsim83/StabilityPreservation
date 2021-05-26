#ifndef INCLUDE_VECTOR_TOOLS_MY_VECTOR_TOOLS_TPP_
#define INCLUDE_VECTOR_TOOLS_MY_VECTOR_TOOLS_TPP_

#include <vector_tools/my_vector_tools.h>

namespace MyVectorTools
{
  using namespace dealii;

  namespace internal
  {
    template <int dim, typename FunctionType>
    void
      assemble_projection(const DoFHandler<dim> &          dof_handler,
                          const AffineConstraints<double> &constraints,
                          const Quadrature<dim> &          quad_rule,
                          const FunctionType &             my_function,
                          TrilinosWrappers::SparseMatrix & system_matrix,
                          TrilinosWrappers::MPI::Vector &  system_rhs,
                          const MPI_Comm & /* mpi_communicator */)
    {
      const FiniteElement<dim> &fe = dof_handler.get_fe();

      using ValueType = decltype(my_function.value(Point<dim>()));

      FEValues<dim> fe_values(fe,
                              quad_rule,
                              update_values | update_quadrature_points |
                                update_JxW_values);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points    = quad_rule.size();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::vector<ValueType> rhs_values(n_q_points);
      // Determine the type ExtractorType from ValueType
      const bool is_value_type_tensor =
        std::is_same<ValueType, Tensor<1, dim>>::value;
      using ExtractorType =
        typename std::conditional<is_value_type_tensor,
                                  typename FEValuesExtractors::Vector,
                                  typename FEValuesExtractors::Scalar>::type;
      const ExtractorType extractor(/* first_(vector_)component */ 0);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              cell_matrix = 0.;
              cell_rhs    = 0.;

              fe_values.reinit(cell);

              my_function.value_list(fe_values.get_quadrature_points(),
                                     rhs_values);

              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const ValueType phi_test =
                        fe_values[extractor].value(i, q_point);

                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          const ValueType phi_trial =
                            fe_values[extractor].value(j, q_point);

                          cell_matrix(i, j) +=
                            phi_test * phi_trial * fe_values.JxW(q_point);
                        }

                      cell_rhs(i) +=
                        rhs_values[q_point] * phi_test * fe_values.JxW(q_point);
                    }
                }
              cell->get_dof_indices(local_dof_indices);
              constraints.distribute_local_to_global(cell_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
            }
        } // ++cell

      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
    }

    template <int dim, typename FunctionType>
    void
      assemble_projection(const DoFHandler<dim> &          dof_handler,
                          const AffineConstraints<double> &constraints,
                          const Quadrature<dim> &          quad_rule,
                          const FunctionType &             my_function,
                          SparseMatrix<double> &           system_matrix,
                          Vector<double> &                 system_rhs)
    {
      const FiniteElement<dim> &fe = dof_handler.get_fe();

      using ValueType = decltype(my_function.value(Point<dim>()));

      FEValues<dim> fe_values(fe,
                              quad_rule,
                              update_values | update_quadrature_points |
                                update_JxW_values);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points    = quad_rule.size();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::vector<ValueType> rhs_values(n_q_points);

      // Determine the type ExtractorType from ValueType
      const bool is_value_type_tensor =
        std::is_same<ValueType, Tensor<1, dim>>::value;
      using ExtractorType =
        typename std::conditional<is_value_type_tensor,
                                  typename FEValuesExtractors::Vector,
                                  typename FEValuesExtractors::Scalar>::type;
      const ExtractorType extractor(/* first_(vector_)component */ 0);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell_matrix = 0.;
          cell_rhs    = 0.;

          fe_values.reinit(cell);

          my_function.value_list(fe_values.get_quadrature_points(), rhs_values);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const ValueType phi_test =
                    fe_values[extractor].value(i, q_point);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      const ValueType phi_trial =
                        fe_values[extractor].value(j, q_point);

                      cell_matrix(i, j) +=
                        phi_test * phi_trial * fe_values.JxW(q_point);
                    }

                  cell_rhs(i) +=
                    rhs_values[q_point] * phi_test * fe_values.JxW(q_point);
                }
            }
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        } // ++cell
    }

    void
      solve(const AffineConstraints<double> &     constraints,
            const IndexSet &                      locally_owned_dofs,
            const TrilinosWrappers::SparseMatrix &system_matrix,
            const TrilinosWrappers::MPI::Vector & system_rhs,
            TrilinosWrappers::MPI::Vector &       vec,
            const MPI_Comm &                      mpi_communicator)
    {
      TrilinosWrappers::MPI::Vector completely_distributed_solution(
        locally_owned_dofs, mpi_communicator);
      SolverControl              solver_control(system_rhs.size(), 1e-12);
      TrilinosWrappers::SolverCG solver(solver_control);

      TrilinosWrappers::PreconditionIdentity                 preconditioner;
      TrilinosWrappers::PreconditionIdentity::AdditionalData data;

      preconditioner.initialize(system_matrix, data);

      solver.solve(system_matrix,
                   completely_distributed_solution,
                   system_rhs,
                   preconditioner);

      constraints.distribute(completely_distributed_solution);
      vec = completely_distributed_solution;
    } // end solve

    void
      solve(const AffineConstraints<double> &constraints,
            const SparseMatrix<double> &     system_matrix,
            const Vector<double> &           system_rhs,
            Vector<double> &                 vec)
    { // Solver

      SolverControl solver_control(system_rhs.size(), 1e-12);
      SolverCG<>    solver(solver_control);

      PreconditionIdentity preconditioner;
      preconditioner.initialize(system_matrix);

      solver.solve(system_matrix, vec, system_rhs, preconditioner);

      constraints.distribute(vec);
    } // end solve

  } // namespace internal

  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////

  template <int dim, typename FunctionType>
  void
    project_on_fe_space(const DoFHandler<dim> &          dof_handler,
                        const AffineConstraints<double> &constraints,
                        const Quadrature<dim> &          quad_rule,
                        const FunctionType &             my_function,
                        TrilinosWrappers::MPI::Vector &  vec,
                        const MPI_Comm &                 mpi_communicator)
  {
    // If element is primitive it is invalid.
    // Also there must not be more than one block.
    // This excludes FE_Systems.
    Assert(dof_handler.get_fe().n_blocks() == 1,
           ExcDimensionMismatch(1, dof_handler.get_fe().n_blocks()));

    // Need info about local indices
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs(),
             locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    TrilinosWrappers::MPI::Vector system_rhs;
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    TrilinosWrappers::SparseMatrix system_matrix;
    {
      DynamicSparsityPattern dsp(locally_relevant_dofs);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
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

    internal::assemble_projection(dof_handler,
                                  constraints,
                                  quad_rule,
                                  my_function,
                                  system_matrix,
                                  system_rhs,
                                  mpi_communicator);

    internal::solve(constraints,
                    locally_owned_dofs,
                    system_matrix,
                    system_rhs,
                    vec,
                    mpi_communicator);
  }

  template <int dim, typename FunctionType>
  void
    project_on_fe_space(const DoFHandler<dim> &          dof_handler,
                        const AffineConstraints<double> &constraints,
                        const Quadrature<dim> &          quad_rule,
                        const FunctionType &             my_function,
                        Vector<double> &                 vec)
  {
    // If element is primitive it is invalid.
    // Also there must not be more than one block.
    // This excludes FE_Systems.
    Assert((!dof_handler.get_fe().is_primitive()), FETools::ExcInvalidFE());
    Assert(dof_handler.get_fe().n_blocks() == 1,
           ExcDimensionMismatch(1, dof_handler.get_fe().n_blocks()));

    // Need info about dofs
    const types::global_dof_index n_dofs = dof_handler.n_dofs();

    Vector<double> system_rhs;
    system_rhs.reinit(n_dofs);

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    {
      DynamicSparsityPattern dsp(n_dofs);
      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp,
                                      constraints,
                                      /*keep_constrained_dofs = */ false);
      sparsity_pattern.copy_from(dsp);
    }
    system_matrix.reinit(sparsity_pattern);

    internal::assemble_projection(dof_handler,
                                  constraints,
                                  quad_rule,
                                  my_function,
                                  system_matrix,
                                  system_rhs);

    internal::solve(constraints, system_matrix, system_rhs, vec);
  }

} // namespace MyVectorTools

#endif /* INCLUDE_VECTOR_TOOLS_MY_VECTOR_TOOLS_TPP_ */
