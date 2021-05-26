#ifndef INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_SCHUR_COMPLEMENT_TPP_
#define INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_SCHUR_COMPLEMENT_TPP_

//#include <linear_algebra/approximate_schur_complement.h>

namespace LinearSolvers
{
  using namespace dealii;


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  ApproximateSchurComplement<BlockMatrixType, VectorType, PreconditionerType>::
    ApproximateSchurComplement(const BlockMatrixType &system_matrix)
    : system_matrix(&system_matrix)
    , preconditioner()
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
    , tmp3(system_matrix.block(1, 1).m())
  {
    typename PreconditionerType::AdditionalData data;
    preconditioner.initialize(system_matrix.block(0, 0), data);
  }

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
    ApproximateSchurComplement<BlockMatrixType,
                               VectorType,
                               PreconditionerType>::vmult(VectorType &      dst,
                                                          const VectorType &src)
      const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    preconditioner.vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    system_matrix->block(1, 1).vmult(tmp3, src);
    dst -= tmp3;
  }

  /*
   * Now the MPI verion.
   */

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  ApproximateSchurComplementMPI<BlockMatrixType,
                                VectorType,
                                PreconditionerType>::
    ApproximateSchurComplementMPI(
      const BlockMatrixType &      system_matrix,
      const std::vector<IndexSet> &owned_partitioning,
      MPI_Comm                     mpi_communicator)
    : system_matrix(&system_matrix)
    , preconditioner()
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
    , tmp3(owned_partitioning[1], mpi_communicator)
  {
    typename PreconditionerType::AdditionalData data;
    preconditioner.initialize(system_matrix.block(0, 0), data);
  }

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
    ApproximateSchurComplementMPI<BlockMatrixType,
                                  VectorType,
                                  PreconditionerType>::vmult(VectorType &dst,
                                                             const VectorType
                                                               &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    preconditioner.vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    system_matrix->block(1, 1).vmult(tmp3, src);
    dst -= tmp3;
  }

} // end namespace LinearSolvers


#endif /* INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_SCHUR_COMPLEMENT_TPP_ */
