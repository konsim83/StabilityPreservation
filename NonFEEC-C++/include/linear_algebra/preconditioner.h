#ifndef HELMHOLTZ_PRECON_H_
#define HELMHOLTZ_PRECON_H_

#include "config.h"
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

namespace LinearSolvers
{
  using namespace dealii;

  /*!
   * @class InnerPreconditioner
   *
   * @brief Encapsulation of preconditioner type for gloabl problems
   *
   * Encapsulation of preconditioner type used for the inner
   * matrix in a Schur complement. Works with MPI
   */
  template <int dim>
  class InnerPreconditioner
  {
  public:
    // Parallell, generic
    //	  using type = LA::MPI::PreconditionAMG;
    using type = LA::MPI::PreconditionILU; /* Turns out to be the best */
    //	  using type = PreconditionIdentity;
  };

  /*!
   * @class LocalInnerPreconditioner
   *
   * @brief Encapsulation of preconditioner type for local problems
   *
   * Inner preconditioner used for local (serial) basis problems
   */
  template <int dim>
  class LocalInnerPreconditioner;

  /*!
   * @class LocalInnerPreconditioner<2>
   *
   * @brief Encapsulation of preconditioner type for local problems in 2D
   */
  template <>
  class LocalInnerPreconditioner<2>
  {
  public:
    using type = SparseDirectUMFPACK;
  };

  /*!
   * @class LocalInnerPreconditioner<3>
   *
   * @brief Encapsulation of preconditioner type for local problems in 3D
   */
  template <>
  class LocalInnerPreconditioner<3>
  {
  public:
    using type = SparseILU<double>;
  };

} // namespace LinearSolvers

#endif /* HELMHOLTZ_PRECON_H_ */
