#ifndef INCLUDE_LINEAR_ALGEBRA_INVERSE_MATRIX_H_
#define INCLUDE_LINEAR_ALGEBRA_INVERSE_MATRIX_H_

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

/*!
 * @namespace LinearSolvers
 *
 * @brief Contains serial and parallelized implementations of solvers and preconditioners.
 */
namespace LinearSolvers
{
  using namespace dealii;

  /*!
   * @class InverseMatrix
   *
   * @brief Implements an iterative inverse
   *
   * Implement the inverse matrix of a given matrix through
   * its action by a preconditioned CG solver. This class also
   * works with MPI.
   *
   * @note The inverse is not constructed explicitly.
   */
  template <typename MatrixType, typename PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    /*!
     * Constructor.
     *
     * @param m
     * @param preconditioner
     */
    InverseMatrix(const MatrixType &        m,
                  const PreconditionerType &preconditioner);

    /*!
     * Matrix-vector multiplication.
     *
     * @param[out] dst
     * @param[in] src
     */
    template <typename VectorType>
    void
      vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Samrt pointer to system matrix.
     */
    const SmartPointer<const MatrixType> matrix;

    /*!
     * Preconditioner.
     */
    const PreconditionerType &preconditioner;
  };

} // end namespace LinearSolvers

#include <linear_algebra/inverse_matrix.tpp>

#endif /* INCLUDE_LINEAR_ALGEBRA_INVERSE_MATRIX_H_ */
