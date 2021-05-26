#ifndef INCLUDE_Q_BASIS_H_
#define INCLUDE_Q_BASIS_H_

// Deal.ii
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

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
#include <equation_data/eqn_exact_solution_lin.h>
#include <equation_data/eqn_rhs.h>
#include <functions/basis_q1.h>
#include <my_other_tools.h>
#include <vector_tools/my_vector_tools.h>

namespace Q
{
  using namespace dealii;

  /*!
   * @class QBasis
   *
   * @brief Class to hold local mutiscale basis in \f$H(\mathrm{grad})\f$.
   *
   * This class is the heart of the multiscale computation. It precomputes the
   * basis on a given cell and assembles the data for the global solver. Once a
   * global solution is computed it takes care of defining the local fine scale
   * solution and writes data.
   */
  class QBasis
  {
  public:
    QBasis() = delete;

    /*!
     * Constructor.
     *
     * @param parameters_ms
     * @param parameter_filename
     * @param global_cell
     * @param first_cell
     * @param local_subdomain
     * @param mpi_communicator
     */
    QBasis(const ParametersMs &                             parameters_ms,
           const std::string &                              parameter_filename,
           typename Triangulation<3>::active_cell_iterator &global_cell,
           CellId                                           first_cell,
           unsigned int                                     local_subdomain,
           MPI_Comm                                         mpi_communicator);

    /*!
     * Copy constructor.
     */
    QBasis(const QBasis &X);

    ~QBasis();

    /*!
     * @brief Run function of the object.
     *
     * Run the computation after object is built.
     */
    void
      run();

    /*!
     * Write out global solution in cell as vtu.
     */
    void
      output_global_solution_in_cell() const;

    /*!
     * Return the multiscale element matrix produced
     * from local basis functions.
     */
    const FullMatrix<double> &
      get_global_element_matrix() const;

    /*!
     * Get the right hand-side that was locally assembled
     * to speed up the global assembly.
     */
    const Vector<double> &
      get_global_element_rhs() const;

    /*!
     * Return filename for local pvtu record.
     */
    const std::string &
      get_filename_global();

    /*!
     * @brief Set global weights.
     *
     * The coarse weights of the global solution determine
     * the local multiscale solution. They must be computed
     * and then set locally to write an output.
     */
    void
      set_global_weights(const std::vector<double> &global_weights);

    /*!
     * Set the output flag to write basis functions to disk as vtu.
     */
    void
      set_output_flag();

  private:
    /*!
     * @brief Set up the grid with a certain number of refinements.
     *
     * Generate a triangulation of \f$[0,1]^{\rm{dim}}\f$ with edges/faces
     * numbered form \f$1,\dots,2\rm{dim}\f$.
     */
    void
      make_grid();

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
     * Assemble the system matrix and the right-hand side.
     */
    void
      assemble_system();

    /*!
     * @brief Assemble the gloabl element matrix and the gobal right-hand side.
     */
    void
      assemble_global_element_matrix();

    /*!
     * @brief Iterative solver.
     *
     * CG-based solver with SSOR-preconditioning.
     */
    void
      solve_direct(unsigned int index_basis);

    /*!
     * @brief Iterative solver.
     *
     * CG-based solver with SSOR-preconditioning.
     */
    void
      solve_iterative(unsigned int index_basis);

    /*!
     * @brief Write basis results to disk.
     *
     * Write basis results to disk in vtu-format.
     */
    void
      output_basis();

    /*!
     * Define the gloabl filename for pvtu-file in global output.
     */
    void
      set_filename_global();

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;


    /*!
     * Parameter structure to hold parsed data.
     */
    ParametersBasis parameters;

    /*!
     * Name of parameter input file.
     */
    const std::string &parameter_filename;

    /*!
     * Local triangulation.
     */
    Triangulation<3> triangulation;

    /*!
     * Finite element system to hold Lagrange elements.
     * This is only used to define the degrees of freedom, not the actual shape
     * functions.
     */
    FE_Q<3> fe;

    DoFHandler<3> dof_handler;

    std::vector<AffineConstraints<double>> constraints_vector;
    std::vector<Point<3>>                  corner_points;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> diffusion_matrix;
    SparseMatrix<double> system_matrix;

    std::string filename_global;

    /*!
     * Solution vector.
     */
    std::vector<Vector<double>> solution_vector;

    /*!
     * Contains the right-hand side.
     */
    Vector<double>
      global_rhs; // this is only for the global assembly (speed-up)

    /*!
     * Contains all parts of the right-hand side needed to
     * solve the linear system..
     */
    Vector<double> system_rhs;

    /*!
     * Holds global multiscale element matrix.
     */
    FullMatrix<double> global_element_matrix;
    bool               is_built_global_element_matrix;

    /*!
     * Holds global multiscale element right-hand side.
     */
    Vector<double> global_element_rhs;

    /*!
     * Weights of multiscale basis functions.
     */
    std::vector<double> global_weights;
    bool                is_set_global_weights;

    /*!
     * Global solution
     */
    Vector<double> global_solution;

    /*!
     * Global cell number.
     */
    const CellId global_cell_id;

    /**
     * Global cell identifier of first cell.
     */
    CellId first_cell;

    /*!
     * Global subdomain number.
     */
    const unsigned int local_subdomain;

    // Geometry info
    double              volume_measure;
    std::vector<double> face_measure;
    std::vector<double> edge_measure;

    /*!
     * Object carries set of local \f$Q_1\f$-basis functions.
     */
    ShapeFun::BasisQ1<3> basis_q1;
  };

} // end namespace Q

#endif /* INCLUDE_Q_BASIS_H_ */
