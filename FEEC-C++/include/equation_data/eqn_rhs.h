#ifndef EQN_RHS_H_
#define EQN_RHS_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

// STL
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

// my headers
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>
#include <equation_data/eqn_exact_solution_lin.h>

namespace EquationData
{
  using namespace dealii;


  /*!
   * @class RightHandSide
   *
   * @brief  Base class for right-hand side.
   *
   * The base class for the right hand side. It inherits from
   * Functions::ParsedFunction<3> so that is can (but does not have to) be
   * parsed from an expression that the user provides.
   */
  class RightHandSide : public Functions::ParsedFunction<3>
  {
  public:
    /*!
     * Constructor.
     *
     * @param n_components
     */
    RightHandSide(unsigned int n_components)
      : Functions::ParsedFunction<3>(n_components)
    {}

    /*!
     * The function must implement a tensor_value_list function for now since in
     * the current version of deal.ii it is not yet possible to parse tensor
     * functions. This feature will be supported from deal.ii 9.2 onwards.
     */
    virtual void
      tensor_value_list(const std::vector<Point<3>> &,
                        std::vector<Tensor<1, 3>> &) const {};
  };



  /*!
   * @class RightHandSideParsed
   *
   * @brief Right-hand side for user provide function expression
   *
   * Right hand side of equation is a parsed function provided by the user.
   */
  class RightHandSideParsed : public RightHandSide
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     * @param n_components
     */
    RightHandSideParsed(const std::string &parameter_filename,
                        unsigned int       n_components);

    /*!
     * Implementation of right hand side. for use with tensors. This is just
     * an interface routine.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      tensor_value_list(const std::vector<Point<3>> &points,
                        std::vector<Tensor<1, 3>> &  values) const override;
  };



  /*!
   * @class RightHandSideExactLin
   *
   * @brief Right hand side used if user chooses to compare an analytic solutions
   *
   * Right hand side of equation is a vectorial function. This class
   * is derived from the class of an abstract solution class and ignores
   * user provided function in the parameter file.
   */
  class RightHandSideExactLin : public RightHandSide,
                                public ExactSolutionLin_Data,
                                public Diffusion_A_Data,
                                public Diffusion_B_Data
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     */
    RightHandSideExactLin(const std::string &parameter_filename);

    /*!
     * Implementation of right hand side of exact solution. Must be given as a
     * tensor.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      tensor_value_list(const std::vector<Point<3>> &points,
                        std::vector<Tensor<1, 3>> &  values) const override;

  private:
    const double pi = numbers::PI;

    /*!
     * Expression for \f$R^T\nabla\times u\f$ if \f$R^T\in SO(3)\f$.
     */
    Tensor<1, 3> R_trans_curl_u;
  };

} // end namespace EquationData

#endif /* EQN_RHS_H_ */
