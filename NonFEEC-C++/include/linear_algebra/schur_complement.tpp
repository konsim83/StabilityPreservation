#ifndef INCLUDE_LINEAR_ALGEBRA_SCHUR_COMPLEMENT_TPP_
#define INCLUDE_LINEAR_ALGEBRA_SCHUR_COMPLEMENT_TPP_

//#include <linear_algebra/schur_complement.h>


namespace LinearSolvers
{
  using namespace dealii;

  /*
   * The serial version
   */

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType,
            typename DoFHandlerType>
  SchurComplement<BlockMatrixType,
                  VectorType,
                  PreconditionerType,
                  DoFHandlerType>::
    SchurComplement(const BlockMatrixType &system_matrix,
                    const InverseMatrix<BlockType, PreconditionerType>
                      &relevant_inverse_matrix)
    : system_matrix(&system_matrix)
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , correct_mean_value(false)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
    , tmp3(system_matrix.block(1, 1).m())
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType,
            typename DoFHandlerType>
  SchurComplement<BlockMatrixType,
                  VectorType,
                  PreconditionerType,
                  DoFHandlerType>::
    SchurComplement(const BlockMatrixType &system_matrix,
                    const InverseMatrix<BlockType, PreconditionerType>
                      &                   relevant_inverse_matrix,
                    const DoFHandlerType &_dof_handler)
    : system_matrix(&system_matrix)
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , dof_handler(_dof_handler.get_triangulation())
    , correct_mean_value(true)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
    , tmp3(system_matrix.block(1, 1).m())
  {
    FEValuesExtractors::Scalar last_component(3);
    const auto &               second_fe(_dof_handler.get_fe());
    ComponentMask last_component_mask(second_fe.component_mask(last_component));
    const auto &  last_fe(second_fe.get_sub_fe(last_component_mask));
    dof_handler.distribute_dofs(last_fe);
  }

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType,
            typename DoFHandlerType>
  void
    SchurComplement<BlockMatrixType,
                    VectorType,
                    PreconditionerType,
                    DoFHandlerType>::vmult(VectorType &      dst,
                                           const VectorType &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    system_matrix->block(1, 1).vmult(tmp3, src);
    dst -= tmp3;

    if (correct_mean_value)
      {
        const double mean_value =
          VectorTools::compute_mean_value(dof_handler, QGauss<3>(1), dst, 0);
        dst.add(-mean_value);
      }
  }

  /*
   * Now the MPI version
   */

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  SchurComplementMPI<BlockMatrixType, VectorType, PreconditionerType>::
    SchurComplementMPI(const BlockMatrixType &system_matrix,
                       const InverseMatrix<BlockType, PreconditionerType>
                         &                          relevant_inverse_matrix,
                       const std::vector<IndexSet> &owned_partitioning,
                       MPI_Comm                     mpi_communicator)
    : system_matrix(&system_matrix)
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
    , tmp3(owned_partitioning[1], mpi_communicator)
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
    SchurComplementMPI<BlockMatrixType, VectorType, PreconditionerType>::vmult(
      VectorType &      dst,
      const VectorType &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    system_matrix->block(1, 1).vmult(tmp3, src);
    dst -= tmp3;
  }

} // end namespace LinearSolvers


#endif /* INCLUDE_LINEAR_ALGEBRA_SCHUR_COMPLEMENT_TPP_ */
