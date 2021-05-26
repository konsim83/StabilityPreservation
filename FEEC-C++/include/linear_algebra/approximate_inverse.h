#ifndef INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_H_
#define INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_H_

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

// STL
#include <memory>

// my headers
#include <config.h>

namespace LinearSolvers
{
  using namespace dealii;

  /*!
   * @class ApproximateInverseMatrix
   *
   * @brief Approximate inverse matrix
   *
   * Approximate inverse matrix through use of preconditioner and a limited
   * number of CG iterations.
   */
  template <typename MatrixType, typename PreconditionerType>
  class ApproximateInverseMatrix : public Subscriptor
  {
  public:
    /*!
     * Constructor.
     *
     * @param m
     * @param preconditioner
     * @param n_iter
     */
    ApproximateInverseMatrix(const MatrixType &        m,
                             const PreconditionerType &preconditioner,
                             const unsigned int        n_iter);

    /*!
     * Matrix vector multiplication. VectorType template can be serial or
     * distributed.
     * @param dst
     * @param src
     */
    template <typename VectorType>
    void
      vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Smart pointer to matrix.
     */
    const SmartPointer<const MatrixType> matrix;

    /*!
     * Pointer to type of preconsitioner.
     */
    const PreconditionerType &preconditioner;

    /*!
     * Maximum number of CG iterations.
     */
    const unsigned int max_iter;
  };

} // end namespace LinearSolvers

#include <linear_algebra/approximate_inverse.tpp>

#endif /* INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_H_ */
