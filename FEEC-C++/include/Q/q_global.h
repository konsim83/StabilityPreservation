#ifndef INCLUDE_Q_GLOBAL_HPP_
#define INCLUDE_Q_GLOBAL_HPP_

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
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/vector.h>
// For distributing the sparsity pattern.
#include <deal.II/lac/sparsity_tools.h>

// Distributed triangulation
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
#include <map>
#include <memory>
#include <vector>

// My Headers
#include <Q/q_basis.h>
#include <Q/q_parameters.h>
#include <Q/q_post_processor.h>
#include <config.h>
#include <equation_data/eqn_boundary_vals.h>
#include <my_other_tools.h>

namespace Q
{
  using namespace dealii;

  /*!
   * @class QMultiscale
   *
   * @brief \f$H(\mathrm{grad})\f$ multiscale solver with Lagrange elements.
   *
   * This class contains a multiscale solver for the weighted 0-form Laplacian
   * with rough coefficients in \f$H(\mathrm{grad})\f$ with Lagrange elements.
   * The solver is MPI parallel and can be used on clusters.
   */
  class QMultiscale
  {
  public:
    /*!
     * Constructor.
     */
    QMultiscale(ParametersMs &     parameters_,
                const std::string &parameter_filename_);

    /*!
     * Destructor.
     */
    ~QMultiscale();

    /*!
     * @brief Run function of the object.
     *
     * Run the computation after object is built.
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
     * Initialize and precompute the basis on each locally owned cell.
     */
    void
      initialize_and_compute_basis();

    /*!
     * @brief Setup sparsity pattern and system matrix.
     *
     * Compute sparsity pattern and reserve memory for the sparse system matrix
     * and a number of right-hand side vectors. Also build a constraint object
     * to take care of Dirichlet boundary conditions.
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
     * @brief Send coarse weights to corresponding local cell.
     *
     * After the coarse (global) weights have been computed they
     * must be set to the local basis object and stored there.
     * This is necessary to write the local multiscale solution.
     */
    void
      send_global_weights_to_cell();

    /*!
     * Write all local multiscale solution (threaded) and
     * a global pvtu-record.
     */
    void
      output_results();

    /*!
     * Collect local file names on all mpi processes to write
     * the global pvtu-record.
     */
    std::vector<std::string>
      collect_filenames_on_mpi_process();

    MPI_Comm mpi_communicator;

    ParametersMs &     parameters;
    const std::string &parameter_filename;

    parallel::distributed::Triangulation<3> triangulation;

    FE_Q<3>       fe;
    DoFHandler<3> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    /*!
     *
     */
    LA::MPI::SparseMatrix system_matrix;

    /*!
     * Solution vector containing weights at the dofs.
     */
    LA::MPI::Vector locally_relevant_solution;

    /*!
     * Contains all parts of the right-hand side needed to
     * solve the linear system.
     */
    LA::MPI::Vector system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    /*!
     * STL Vector holding basis functions for each coarse cell.
     */
    std::map<CellId, QBasis> cell_basis_map;

    /*!
     * Identifier for first cell in global triangulation.
     */
    CellId first_cell;
  };

} // end namespace Q


#endif /* INCLUDE_Q_GLOBAL_HPP_ */
