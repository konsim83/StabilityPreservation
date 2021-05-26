#include <vector_tools/my_vector_tools.tpp>

namespace MyVectorTools
{
  using namespace dealii;

  namespace internal
  {
    ////////////////////////
    // explicit instantiations
    ////////////////////////

    template void
      assemble_projection<2, TensorFunction<1, 2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const TensorFunction<1, 2> &     my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    template void
      assemble_projection<3, TensorFunction<1, 3>>(
        const DoFHandler<3> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<3> &            quad_rule,
        const TensorFunction<1, 3> &     my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    template void
      assemble_projection<2, TensorFunction<1, 2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const TensorFunction<1, 2> &     my_function,
        SparseMatrix<double> &           system_matrix,
        Vector<double> &                 system_rhs);

    template void
      assemble_projection<3, TensorFunction<1, 3>>(
        const DoFHandler<3> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<3> &            quad_rule,
        const TensorFunction<1, 3> &     my_function,
        SparseMatrix<double> &           system_matrix,
        Vector<double> &                 system_rhs);

    ////////////////////////

    template void
      assemble_projection<2, Function<2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const Function<2> &              my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    template void
      assemble_projection<2, Function<2>>(
        const DoFHandler<2> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<2> &            quad_rule,
        const Function<2> &              my_function,
        SparseMatrix<double> &           system_matrix,
        Vector<double> &                 system_rhs);

    template void
      assemble_projection<3, Function<3>>(
        const DoFHandler<3> &            dof_handler,
        const AffineConstraints<double> &constraints,
        const Quadrature<3> &            quad_rule,
        const Function<3> &              my_function,
        TrilinosWrappers::SparseMatrix & system_matrix,
        TrilinosWrappers::MPI::Vector &  system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    template void
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

  ////////////////////////
  // explicit instantiations
  ////////////////////////

  template void
    project_on_fe_space<2, TensorFunction<1, 2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const TensorFunction<1, 2> &     my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  template void
    project_on_fe_space<2, TensorFunction<1, 2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const TensorFunction<1, 2> &     my_function,
      Vector<double> &                 vec);

  template void
    project_on_fe_space<3, TensorFunction<1, 3>>(
      const DoFHandler<3> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<3> &            quad_rule,
      const TensorFunction<1, 3> &     my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  template void
    project_on_fe_space<3, TensorFunction<1, 3>>(
      const DoFHandler<3> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<3> &            quad_rule,
      const TensorFunction<1, 3> &     my_function,
      Vector<double> &                 vec);

  ////////////////////////

  template void
    project_on_fe_space<2, Function<2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const Function<2> &              my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  template void
    project_on_fe_space<2, Function<2>>(
      const DoFHandler<2> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<2> &            quad_rule,
      const Function<2> &              my_function,
      Vector<double> &                 vec);

  template void
    project_on_fe_space<3, Function<3>>(
      const DoFHandler<3> &            dof_handler,
      const AffineConstraints<double> &constraints,
      const Quadrature<3> &            quad_rule,
      const Function<3> &              my_function,
      TrilinosWrappers::MPI::Vector &  vec,
      const MPI_Comm &                 mpi_communicator);

  template void
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
    template void
      assemble_projection<3, EquationData::ExactSolutionLin_B_div>(
        const DoFHandler<3> &                       dof_handler,
        const AffineConstraints<double> &           constraints,
        const Quadrature<3> &                       quad_rule,
        const EquationData::ExactSolutionLin_B_div &my_function,
        TrilinosWrappers::SparseMatrix &            system_matrix,
        TrilinosWrappers::MPI::Vector &             system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    template void
      assemble_projection<3, EquationData::ExactSolutionLin_B_div>(
        const DoFHandler<3> &                       dof_handler,
        const AffineConstraints<double> &           constraints,
        const Quadrature<3> &                       quad_rule,
        const EquationData::ExactSolutionLin_B_div &my_function,
        SparseMatrix<double> &                      system_matrix,
        Vector<double> &                            system_rhs);

    template void
      assemble_projection<3, EquationData::ExactSolutionLin>(
        const DoFHandler<3> &                 dof_handler,
        const AffineConstraints<double> &     constraints,
        const Quadrature<3> &                 quad_rule,
        const EquationData::ExactSolutionLin &my_function,
        TrilinosWrappers::SparseMatrix &      system_matrix,
        TrilinosWrappers::MPI::Vector &       system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    template void
      assemble_projection<3, EquationData::ExactSolutionLin>(
        const DoFHandler<3> &                 dof_handler,
        const AffineConstraints<double> &     constraints,
        const Quadrature<3> &                 quad_rule,
        const EquationData::ExactSolutionLin &my_function,
        SparseMatrix<double> &                system_matrix,
        Vector<double> &                      system_rhs);

    template void
      assemble_projection<3, EquationData::ExactSolutionLin_A_curl>(
        const DoFHandler<3> &                        dof_handler,
        const AffineConstraints<double> &            constraints,
        const Quadrature<3> &                        quad_rule,
        const EquationData::ExactSolutionLin_A_curl &my_function,
        TrilinosWrappers::SparseMatrix &             system_matrix,
        TrilinosWrappers::MPI::Vector &              system_rhs,
        const MPI_Comm & /* mpi_communicator */);

    template void
      assemble_projection<3, EquationData::ExactSolutionLin_A_curl>(
        const DoFHandler<3> &                        dof_handler,
        const AffineConstraints<double> &            constraints,
        const Quadrature<3> &                        quad_rule,
        const EquationData::ExactSolutionLin_A_curl &my_function,
        SparseMatrix<double> &                       system_matrix,
        Vector<double> &                             system_rhs);

  } // namespace internal

  template void
    project_on_fe_space<3, EquationData::ExactSolutionLin_B_div>(
      const DoFHandler<3> &                       dof_handler,
      const AffineConstraints<double> &           constraints,
      const Quadrature<3> &                       quad_rule,
      const EquationData::ExactSolutionLin_B_div &my_function,
      TrilinosWrappers::MPI::Vector &             vec,
      const MPI_Comm &                            mpi_communicator);

  template void
    project_on_fe_space<3, EquationData::ExactSolutionLin_B_div>(
      const DoFHandler<3> &                       dof_handler,
      const AffineConstraints<double> &           constraints,
      const Quadrature<3> &                       quad_rule,
      const EquationData::ExactSolutionLin_B_div &my_function,
      Vector<double> &                            vec);

  template void
    project_on_fe_space<3, EquationData::ExactSolutionLin>(
      const DoFHandler<3> &                 dof_handler,
      const AffineConstraints<double> &     constraints,
      const Quadrature<3> &                 quad_rule,
      const EquationData::ExactSolutionLin &my_function,
      TrilinosWrappers::MPI::Vector &       vec,
      const MPI_Comm &                      mpi_communicator);

  template void
    project_on_fe_space<3, EquationData::ExactSolutionLin>(
      const DoFHandler<3> &                 dof_handler,
      const AffineConstraints<double> &     constraints,
      const Quadrature<3> &                 quad_rule,
      const EquationData::ExactSolutionLin &my_function,
      Vector<double> &                      vec);

  template void
    project_on_fe_space<3, EquationData::ExactSolutionLin_A_curl>(
      const DoFHandler<3> &                        dof_handler,
      const AffineConstraints<double> &            constraints,
      const Quadrature<3> &                        quad_rule,
      const EquationData::ExactSolutionLin_A_curl &my_function,
      TrilinosWrappers::MPI::Vector &              vec,
      const MPI_Comm &                             mpi_communicator);

  template void
    project_on_fe_space<3, EquationData::ExactSolutionLin_A_curl>(
      const DoFHandler<3> &                        dof_handler,
      const AffineConstraints<double> &            constraints,
      const Quadrature<3> &                        quad_rule,
      const EquationData::ExactSolutionLin_A_curl &my_function,
      Vector<double> &                             vec);

} // namespace MyVectorTools
