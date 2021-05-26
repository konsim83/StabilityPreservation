#ifndef INCLUDE_LINEAR_ALGEBRA_INVERSE_MATRIX_TPP_
#define INCLUDE_LINEAR_ALGEBRA_INVERSE_MATRIX_TPP_

//#include <linear_algebra/inverse_matrix.h>

namespace LinearSolvers
{
  using namespace dealii;

  template <typename MatrixType, typename PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &        m,
    const PreconditionerType &preconditioner)
    : matrix(&m)
    , preconditioner(preconditioner)
  {}

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  void
    InverseMatrix<MatrixType, PreconditionerType>::vmult(
      VectorType &      dst,
      const VectorType &src) const
  {
    SolverControl solver_control(std::max(static_cast<std::size_t>(src.size()),
                                          static_cast<std::size_t>(1000)),
                                 1e-6 * src.l2_norm());
    SolverGMRES<VectorType> local_solver(solver_control);

    dst = 0;

    try
      {
        local_solver.solve(*matrix, dst, src, preconditioner);
      }
    catch (std::exception &e)
      {
        Assert(false, ExcMessage(e.what()));
      }
  }

} // end namespace LinearSolvers

#endif /* INCLUDE_LINEAR_ALGEBRA_INVERSE_MATRIX_TPP_ */
