#ifndef Q_NED_REF_H_
#define Q_NED_REF_H_

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/vector.h>
// For distributing the sparsity pattern.
#include <deal.II/lac/sparsity_tools.h>

// Distributed triangulation
#include <deal.II/base/timer.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// C++
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

// my headers
#include <Q_Ned/q_ned_parameters.h>
#include <Q_Ned/q_ned_post_processor.h>
#include <config.h>
#include <equation_data/eqn_boundary_vals.h>
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>
#include <equation_data/eqn_coeff_R.h>
#include <equation_data/eqn_exact_solution_lin.h>
#include <equation_data/eqn_rhs.h>
#include <functions/concatinate_functions.h>
#include <linear_algebra/approximate_inverse.h>
#include <linear_algebra/approximate_schur_complement.h>
#include <linear_algebra/inverse_matrix.h>
#include <linear_algebra/preconditioner.h>
#include <linear_algebra/schur_complement.h>
#include <my_other_tools.h>
#include <vector_tools/my_vector_tools.h>

/*!
 * @namespace QNed
 *
 * @brief Namespace for \f$H(\mathrm{grad})\f$-\f$H(\mathrm{curl})\f$ problems with conformal multiscale elements.
 */
namespace QNed
{
  using namespace dealii;

  /*!
   * @class QNedStd
   *
   * @brief \f$H(\mathrm{grad})\f$-\f$H(\mathrm{curl})\f$ problem solver with Lagrange-Nedelec pairings.
   *
   * This class contains a standard solver for the weighted 1-form Laplacian
   * with rough coefficients in \f$H(\mathrm{grad})\f$-\f$H(\mathrm{curl})\f$
   * with Lagrange-Nedelec pairings. The solver is MPI parallel and can be
   * used on clusters.
   */
  class QNedStd
  {
  public:
    QNedStd() = delete;

    /*!
     * Constructor.
     *
     * @param parameters_
     * @param parameter_filename_
     */
    QNedStd(ParametersStd &parameters_, const std::string &parameter_filename_);

    ~QNedStd();

    /*!
     * Solve standard mixed problem with Lagrange-Nedelec element
     * pairing.
     */
    void
      run();

  private:
    /*!
     * Set up grid.
     */
    void
      setup_grid();

    /*!
     * Set up system matrix.
     */
    void
      setup_system_matrix();

    /*!
     * Setup constraints.
     */
    void
      setup_constraints();

    /*!
     * Assemble the system matrix.
     */
    void
      assemble_system();

    /*!
     * Sparse direct MUMPS for block systems.
     *
     * * @note This will throw an exception since this is not implemented in deal.ii v9.1.1 for BlockSparseMatrix.
     */
    void
      solve_direct();

    /*!
     * Schur complement solver uses a preconditioned approximate Schur
     * complement solver.
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
     * If the user decides to use an exact solution as a ground truth then the
     * solution must be projected onto the Nedelec-Raviart-Thomas space.
     */
    void
      write_exact_solution();

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
     * Finite element system to hold Lagrange-Nedelec element pairing.
     */
    FESystem<3> fe;

    // Modified DoFHandler
    DoFHandler<3> dof_handler;

    IndexSet              locally_relevant_dofs;
    std::vector<IndexSet> owned_partitioning;
    std::vector<IndexSet> relevant_partitioning;

    // Constraint matrix holds boundary conditions
    AffineConstraints<double> constraints;

    /*!
     * Distributed system matrix.
     */
    LA::MPI::BlockSparseMatrix system_matrix;

    /*!
     * Solution vector containing weights at the dofs.
     */
    LA::MPI::BlockVector locally_relevant_solution;

    /*!
     * Exact solution vector containing weights at the dofs.
     */
    LA::MPI::BlockVector locally_relevant_exact_solution;

    /*!
     * Contains all parts of the right-hand side needed to
     * solve the linear system.
     */
    LA::MPI::BlockVector system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    std::shared_ptr<typename LinearSolvers::InnerPreconditioner<3>::type>
      inner_schur_preconditioner;
  };

} // end namespace QNed

#endif /* Q_NED_REF_H_ */
