#ifndef INCLUDE_Q_H_
#define INCLUDE_Q_H_


/* ***************
 * Deal.ii
 * ***************
 */
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
// For distributing the sparsity pattern.
#include <deal.II/lac/sparsity_tools.h>

// Distributed triangulation
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
/* ***************
 * Deal.ii
 * ***************
 * */

// STL
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

// my headers
#include <Q/q_parameters.h>
#include <Q/q_post_processor.h>
#include <config.h>
#include <equation_data/eqn_boundary_vals.h>
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_R.h>
#include <equation_data/eqn_rhs.h>
#include <my_other_tools.h>
#include <vector_tools/my_vector_tools.h>

/*!
 * @namespace Q
 * @brief Contains implementation of the main object
 * and all functions to solve a
 * Dirichlet-Neumann problem on a unit square.
 */
namespace Q
{
  using namespace dealii;


  /*!
   * @class QStd
   *
   * @brief \f$H(\mathrm{grad})\f$ multiscale solver with Lagrange elements.
   *
   * This class contains a multiscale solver for the weighted 0-form Laplacian
   * with rough coefficients in \f$H(\mathrm{grad})\f$ with Lagrange elements.
   * The solver is MPI parallel and can be used on clusters.
   */
  class QStd
  {
  public:
    /*!
     * Delete default constructor.
     */
    QStd() = delete;

    /*!
     * Constructor.
     *
     * @param parameters_
     * @param parameter_filename_
     */
    QStd(ParametersStd &parameters_, const std::string &parameter_filename_);

    ~QStd();

    /*!
     * Solve standard mixed problem with modified Lagrange elements.
     */
    void
      run();

  private:
    /*!
     * Set up grid.
     */
    void
      make_grid();

    /*!
     * Setup sparsity pattern and system matrix.
     */
    void
      setup_system();

    /*!
     * Assemble the system matrix.
     */
    void
      assemble_system();

    /*!
     * @brief Sparse direct solver.
     *
     * Apply parallel sparse direct MUMPS through the Amesos2 package of
     * Trilinos.
     */
    void
      solve_direct();

    /*!
     * @brief Iterative solver.
     *
     * CG-based solver with AMG-preconditioning.
     */
    void
      solve_iterative();

    /*!
     * @brief Transfer solution to a finer grid.
     *
     * In order to be able to compare solutions on different refinement levels
     * we need to transfer coarse solutions to finer grids.
     */
    void
      transfer_solution();

    /*!
     * Write *.vtu output and a pvtu-record that collects the vtu-files.
     */
    void
      output_results() const;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Parameter structure to hold parsed data.
     */
    ParametersStd &parameters;

    /*!
     * Name of parameter input file.
     */
    const std::string &parameter_filename;

    /*!
     * Distributed triangulation.
     */
    parallel::distributed::Triangulation<3> triangulation;

    /*!
     * Standard Lagrange elements.
     */
    FE_Q<3> fe;

    DoFHandler<3> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    /*!
     * Distributed system matrix.
     */
    LA::MPI::SparseMatrix system_matrix;

    /*!
     * Exact solution vector containing weights at the dofs.
     */
    LA::MPI::Vector locally_relevant_solution;

    /*!
     * Contains all parts of the right-hand side needed to
     * solve the linear system.
     */
    LA::MPI::Vector system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };

} // end namespace Q


#endif /* INCLUDE_Q_H_ */
