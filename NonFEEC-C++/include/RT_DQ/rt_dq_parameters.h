#ifndef INCLUDE_RT_DQ_PARAMETERS_H_
#define INCLUDE_RT_DQ_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/cell_id.h>

#include <fstream>
#include <iostream>
#include <vector>

/**
 * Namespace for RT-DQ problems.
 */
namespace RTDQ
{
  using namespace dealii;

  struct ParametersStd
  {
    ParametersStd(const std::string &parameter_filename);

    static void
      declare_parameters(ParameterHandler &prm);
    void
      parse_parameters(ParameterHandler &prm);

    const bool degree = 0;

    bool compute_solution;
    bool is_pure_neumann;
    bool verbose;
    bool use_direct_solver; /* This is often better for 2D problems. */
    bool renumber_dofs;     /* Reduce bandwidth in either system component */

    unsigned int n_refine;
    int          transfer_to_level;

    const bool is_laplace = true;

    std::string filename_output;
    std::string dirname_output;
  };


  struct ParametersMs
  {
    ParametersMs(const std::string &parameter_filename);

    static void
      declare_parameters(ParameterHandler &prm);
    void
      parse_parameters(ParameterHandler &prm);

    bool compute_solution;
    bool is_pure_neumann;
    bool verbose;
    bool verbose_basis;
    bool use_direct_solver;       /* This is often better for 2D problems. */
    bool use_direct_solver_basis; /* This is often better for 2D problems. */
    bool renumber_dofs;  /* Reduce bandwidth in either system component */
    bool prevent_output; /* Prevent output of first cell's basis */

    unsigned int n_refine_global;
    unsigned int n_refine_local;

    const bool is_laplace = true;

    std::string filename_output;
    std::string dirname_output;
  };


  struct ParametersBasis
  {
    ParametersBasis(const ParametersMs &param_ms);
    ParametersBasis(
      const ParametersBasis &other); // This the the copy constructor

    void
      set_output_flag(CellId local_cell_id, CellId first_cell);

    const unsigned int degree     = 0;
    const bool         set_to_std = false;

    bool verbose;
    bool use_direct_solver; /* This is often better for 2D problems. */
    bool renumber_dofs;     /* Reduce bandwidth in either system component */

    bool prevent_output;
    bool output_flag;

    unsigned int n_refine_global;
    unsigned int n_refine_local;

    const bool is_laplace            = true;
    const bool fast_constraint_setup = false; // false is safer but way slower

    std::string filename_global;
    std::string dirname_output;
  };

} // namespace RTDQ

#endif /* INCLUDE_RT_DQ_PARAMETERS_H_ */
