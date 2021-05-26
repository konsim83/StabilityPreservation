#ifndef INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_SCHUR_COMPLEMENT_H_
#define INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_SCHUR_COMPLEMENT_H_

#include "config.h"
#include <deal.II/base/subscriptor.h>
#include <linear_algebra/inverse_matrix.h>

#include <memory>
#include <vector>

namespace LinearSolvers
{
  using namespace dealii;

  /*
   * Serial version.
   */

  /*!
   * @class ApproximateSchurComplement
   *
   * @brief Implements a serial approximate Schur complement
   *
   * Implements a serial approximate Schur complement through the use of a
   * preconditioner for the inner inverse matrix, i.e., if we want to solve
   * \f{eqnarray}{
   *	\left(
   *	\begin{array}{cc}
   *		A & B^T \\
   *		B & C
   *	\end{array}
   *	\right)
   *	\left(
   *	\begin{array}{c}
   *		\sigma \\
   *		u
   *	\end{array}
   *	\right)
   *	=
   *	\left(
   *	\begin{array}{c}
   *		0 \\
   *		u
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$A\f$ is invertible then we choose a preconditioner
   *\f$P_A\f$ and define the approximate Schur complement as \f{eqnarray}{
   *\tilde S = C - BP_A^{-1}B^T \f} to solve for \f$u\f$.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class ApproximateSchurComplement : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor.
     *
     * @param system_matrix
     */
    ApproximateSchurComplement(const BlockMatrixType &system_matrix);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
      vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Samrt pointer to system matrix.
     */
    const SmartPointer<const BlockMatrixType> system_matrix;

    /*!
     * Preconditioner.
     */
    PreconditionerType preconditioner;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2, tmp3;
  };

  /*
   * MPI version.
   */

  /*!
   * @class ApproximateSchurComplementMPI
   *
   * @brief Implements a MPI parallel approximate Schur complement
   *
   * Like the ApproximateSchurComplement class just MPI parallel.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class ApproximateSchurComplementMPI : public Subscriptor
  {
  private:
    /*!
     * Typedef for convenience.
     */
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor.
     *
     * @param system_matrix
     * @param owned_partitioning
     * @param mpi_communicator
     */
    ApproximateSchurComplementMPI(
      const BlockMatrixType &      system_matrix,
      const std::vector<IndexSet> &owned_partitioning,
      MPI_Comm                     mpi_communicator);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
      vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Samrt pointer to system matrix.
     */
    const SmartPointer<const BlockMatrixType> system_matrix;

    /*!
     * Preconditioner.
     */
    PreconditionerType preconditioner;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*!
     * Relevant MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable distributed types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2, tmp3;
  };

} // end namespace LinearSolvers

#include <linear_algebra/approximate_schur_complement.tpp>

#endif /* INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_SCHUR_COMPLEMENT_H_ */
