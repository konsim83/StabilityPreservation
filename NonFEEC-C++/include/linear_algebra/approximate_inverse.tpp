#ifndef INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_TPP_
#define INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_TPP_

//#include <linear_algebra/approximate_inverse.h>

namespace LinearSolvers
{
  using namespace dealii;

  template <typename MatrixType, typename PreconditionerType>
  ApproximateInverseMatrix<MatrixType, PreconditionerType>::
    ApproximateInverseMatrix(const MatrixType &        m,
                             const PreconditionerType &preconditioner,
                             const unsigned int        n_iter)
    : matrix(&m)
    , preconditioner(preconditioner)
    , max_iter(n_iter)
  {}

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  void
    ApproximateInverseMatrix<MatrixType, PreconditionerType>::vmult(
      VectorType &      dst,
      const VectorType &src) const
  {
    SolverControl solver_control(/* max_iter */ max_iter, 1e-6 * src.l2_norm());
    SolverCG<VectorType> local_solver(solver_control);

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

#endif /* INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_TPP_ */
