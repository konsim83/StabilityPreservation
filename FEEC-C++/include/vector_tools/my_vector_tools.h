#ifndef INCLUDE_VECTOR_TOOLS_MY_VECTOR_TOOLS_H_
#define INCLUDE_VECTOR_TOOLS_MY_VECTOR_TOOLS_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// STL
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

// my headers
#include <equation_data/eqn_exact_solution_lin.h>

/*
 * Necessary for external instantiation declaration.
 * Not good style but works so far.
 */

/*!
 * @namespace MyVectorTools
 *
 * @brief Extension of Vector tools namespace contains functions
 * not implemented in deal.ii yet.
 */
namespace MyVectorTools
{
  using namespace dealii;

  namespace internal
  {
    /*!
     * Parallel assembly function for non-matrix-free projections onto fe
     * space.
     *
     * @param dof_handler
     * @param constraints
     * @param quad_rule
     * @param my_function
     * @param system_matrix
     * @param system_rhs
     * @param mpi_communicator
     */
    template <int dim, typename FunctionType>
    void
      assemble_projection(const DoFHandler<dim> &          dof_handler,
                          const AffineConstraints<double> &constraints,
                          const Quadrature<dim> &          quad_rule,
                          const FunctionType &             my_function,
                          TrilinosWrappers::SparseMatrix & system_matrix,
                          TrilinosWrappers::MPI::Vector &  system_rhs,
                          const MPI_Comm &                 mpi_communicator);

    /*!
     * Serial assembly function for non-matrix-free projections onto fe
     * space.
     *
     * @param dof_handler
     * @param constraints
     * @param quad_rule
     * @param my_function
     * @param system_matrix
     * @param system_rhs
     */
    template <int dim, typename FunctionType>
    void
      assemble_projection(const DoFHandler<dim> &          dof_handler,
                          const AffineConstraints<double> &constraints,
                          const Quadrature<dim> &          quad_rule,
                          const FunctionType &             my_function,
                          SparseMatrix<double> &           system_matrix,
                          Vector<double> &                 system_rhs);

    /*!
     * Parallel solver for projection problems. These problems are not
     * too hard so the solvers are quite standard CG solvers without
     * preconditioning.
     *
     * @param constraints
     * @param locally_owned_dofs
     * @param system_matrix
     * @param system_rhs
     * @param vec
     * @param mpi_communicator
     */
    void
      solve(const AffineConstraints<double> &     constraints,
            const IndexSet &                      locally_owned_dofs,
            const TrilinosWrappers::SparseMatrix &system_matrix,
            const TrilinosWrappers::MPI::Vector & system_rhs,
            TrilinosWrappers::MPI::Vector &       vec,
            const MPI_Comm &                      mpi_communicator);

    /*!
     * Serial solver for projection problems. These problems are not
     * too hard so the solvers are quite standard CG solvers without
     * preconditioning.
     *
     * @param constraints
     * @param system_matrix
     * @param system_rhs
     * @param vec
     */
    void
      solve(const AffineConstraints<double> &constraints,
            const SparseMatrix<double> &     system_matrix,
            const Vector<double> &           system_rhs,
            Vector<double> &                 vec);

    /*!
     * Fast dynamic cast. Be careful, this is not safe.
     *
     * @param src
     * @return
     */
    template <typename Dest, typename Src>
    Dest *
      most_derived_fast_dynamic_cast(Src *src)
    {
      if (typeid(*src) == typeid(Dest))
        {
          return static_cast<Dest *>(src);
        }
      throw std::bad_cast();
    }

  } // namespace internal

  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////

  /*!
   * Parallel projection method on finite element space. This works
   * also for some elements that do not work in deal.ii for distributed
   * vectors.
   *
   * @param dof_handler
   * @param constraints
   * @param quad_rule
   * @param my_function
   * @param vec
   * @param mpi_communicator
   */
  template <int dim, typename FunctionType>
  void
    project_on_fe_space(const DoFHandler<dim> &          dof_handler,
                        const AffineConstraints<double> &constraints,
                        const Quadrature<dim> &          quad_rule,
                        const FunctionType &             my_function,
                        TrilinosWrappers::MPI::Vector &  vec,
                        const MPI_Comm &                 mpi_communicator);
  /*!
   * Serial projection method on finite element space. There may be
   * some redundancy with deal.ii.
   *
   * @param dof_handler
   * @param constraints
   * @param quad_rule
   * @param my_function
   * @param vec
   */
  template <int dim, typename FunctionType>
  void
    project_on_fe_space(const DoFHandler<dim> &          dof_handler,
                        const AffineConstraints<double> &constraints,
                        const Quadrature<dim> &          quad_rule,
                        const FunctionType &             my_function,
                        Vector<double> &                 vec);

} // namespace MyVectorTools

/*
 * External Instatiations
 */
namespace MyVectorTools
{
  using namespace dealii;

  namespace internal
  {
    extern template void
      assemble_projection<2, TensorFunction<1, 2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const TensorFunction<1, 2> &     my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    extern template void
      assemble_projection<3, TensorFunction<1, 3>>(
        const DoFHandler<3> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<3> &            quad_rule,
        const TensorFunction<1, 3> &     my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    extern template void
      assemble_projection<2, TensorFunction<1, 2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const TensorFunction<1, 2> &     my_function,
        SparseMatrix<double> &           system_matrix,
        Vector<double> &                 system_rhs);

    extern template void
      assemble_projection<3, TensorFunction<1, 3>>(
        const DoFHandler<3> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<3> &            quad_rule,
        const TensorFunction<1, 3> &     my_function,
        SparseMatrix<double> &           system_matrix,
        Vector<double> &                 system_rhs);

    ////////////////////////

    extern template void
      assemble_projection<2, Function<2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const Function<2> &              my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    extern template void
      assemble_projection<2, Function<2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const Function<2> &              my_function,
        SparseMatrix<double> &           system_matrix,
        Vector<double> &                 system_rhs);

    extern template void
      assemble_projection<3, Function<3>>(
        const DoFHandler<3> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<3> &            quad_rule,
        const Function<3> &              my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    extern template void
      assemble_projection<3, Function<3>>(
        const DoFHandler<3> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<3> &            quad_rule,
        const Function<3> &              my_function,
        SparseMatrix<double> &           system_matrix,
        Vector<double> &                 system_rhs);

  } // namespace internal

  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////

  extern template void
    project_on_fe_space<2, TensorFunction<1, 2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const TensorFunction<1, 2> &     my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  extern template void
    project_on_fe_space<2, TensorFunction<1, 2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const TensorFunction<1, 2> &     my_function,
      Vector<double> &                 vec);

  extern template void
    project_on_fe_space<3, TensorFunction<1, 3>>(
      const DoFHandler<3> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<3> &            quad_rule,
      const TensorFunction<1, 3> &     my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  extern template void
    project_on_fe_space<3, TensorFunction<1, 3>>(
      const DoFHandler<3> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<3> &            quad_rule,
      const TensorFunction<1, 3> &     my_function,
      Vector<double> &                 vec);

  ////////////////////////

  extern template void
    project_on_fe_space<2, Function<2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const Function<2> &              my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  extern template void
    project_on_fe_space<2, Function<2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const Function<2> &              my_function,
      Vector<double> &                 vec);

  extern template void
    project_on_fe_space<3, Function<3>>(
      const DoFHandler<3> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<3> &            quad_rule,
      const Function<3> &              my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  extern template void
    project_on_fe_space<3, Function<3>>(
      const DoFHandler<3> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<3> &            quad_rule,
      const Function<3> &              my_function,
      Vector<double> &                 vec);

  ////////////////////////
  ////////////////////////
  ////////////////////////

  namespace internal
  {
    extern template void
      assemble_projection<3, EquationData::ExactSolutionLin_B_div>(
        const DoFHandler<3> &                       dof_handler,
        const AffineConstraints<double> &           constraints,
        const Quadrature<3> &                       quad_rule,
        const EquationData::ExactSolutionLin_B_div &my_function,
        TrilinosWrappers::SparseMatrix &            system_matrix,
        TrilinosWrappers::MPI::Vector &             system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    extern template void
      assemble_projection<3, EquationData::ExactSolutionLin>(
        const DoFHandler<3> &                 dof_handler,
        const AffineConstraints<double> &     constraints,
        const Quadrature<3> &                 quad_rule,
        const EquationData::ExactSolutionLin &my_function,
        TrilinosWrappers::SparseMatrix &      system_matrix,
        TrilinosWrappers::MPI::Vector &       system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    extern template void
      assemble_projection<3, EquationData::ExactSolutionLin>(
        const DoFHandler<3> &                 dof_handler,
        const AffineConstraints<double> &     constraints,
        const Quadrature<3> &                 quad_rule,
        const EquationData::ExactSolutionLin &my_function,
        SparseMatrix<double> &                system_matrix,
        Vector<double> &                      system_rhs);

    extern template void
      assemble_projection<3, EquationData::ExactSolutionLin_A_curl>(
        const DoFHandler<3> &                        dof_handler,
        const AffineConstraints<double> &            constraints,
        const Quadrature<3> &                        quad_rule,
        const EquationData::ExactSolutionLin_A_curl &my_function,
        TrilinosWrappers::SparseMatrix &             system_matrix,
        TrilinosWrappers::MPI::Vector &              system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    extern template void
      assemble_projection<3, EquationData::ExactSolutionLin_A_curl>(
        const DoFHandler<3> &                        dof_handler,
        const AffineConstraints<double> &            constraints,
        const Quadrature<3> &                        quad_rule,
        const EquationData::ExactSolutionLin_A_curl &my_function,
        SparseMatrix<double> &                       system_matrix,
        Vector<double> &                             system_rhs);

  } // namespace internal

  extern template void
    project_on_fe_space<3, EquationData::ExactSolutionLin_B_div>(
      const DoFHandler<3> &                       dof_handler,
      const AffineConstraints<double> &           constraints,
      const Quadrature<3> &                       quad_rule,
      const EquationData::ExactSolutionLin_B_div &my_function,
      TrilinosWrappers::MPI::Vector &             vec,
      const MPI_Comm &                            mpi_communicator);

  extern template void
    project_on_fe_space<3, EquationData::ExactSolutionLin>(
      const DoFHandler<3> &                 dof_handler,
      const AffineConstraints<double> &     constraints,
      const Quadrature<3> &                 quad_rule,
      const EquationData::ExactSolutionLin &my_function,
      TrilinosWrappers::MPI::Vector &       vec,
      const MPI_Comm &                      mpi_communicator);

  extern template void
    project_on_fe_space<3, EquationData::ExactSolutionLin>(
      const DoFHandler<3> &                 dof_handler,
      const AffineConstraints<double> &     constraints,
      const Quadrature<3> &                 quad_rule,
      const EquationData::ExactSolutionLin &my_function,
      Vector<double> &                      vec);

  extern template void
    project_on_fe_space<3, EquationData::ExactSolutionLin_A_curl>(
      const DoFHandler<3> &                        dof_handler,
      const AffineConstraints<double> &            constraints,
      const Quadrature<3> &                        quad_rule,
      const EquationData::ExactSolutionLin_A_curl &my_function,
      TrilinosWrappers::MPI::Vector &              vec,
      const MPI_Comm &                             mpi_communicator);

  extern template void
    project_on_fe_space<3, EquationData::ExactSolutionLin_A_curl>(
      const DoFHandler<3> &                        dof_handler,
      const AffineConstraints<double> &            constraints,
      const Quadrature<3> &                        quad_rule,
      const EquationData::ExactSolutionLin_A_curl &my_function,
      Vector<double> &                             vec);

} // namespace MyVectorTools

#endif /* INCLUDE_VECTOR_TOOLS_MY_VECTOR_TOOLS_H_ */
