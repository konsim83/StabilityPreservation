#ifndef INCLUDE_EQUATION_DATA_EQN_EXACT_SOLUTION_LIN_H_
#define INCLUDE_EQUATION_DATA_EQN_EXACT_SOLUTION_LIN_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace EquationData
{
  using namespace dealii;

  /*!
   * @class ExactSolutionLin_Data
   *
   * @brief Data class to construct an analytic solution.
   *
   * This class contains only data members read from a parameter file to
   * generate equation data for an analytic solution if \f$u\f$ is a vector
   * proxy, i.e., a 1-form or a 2-form. The analytic solution is \f{eqnarray}{
   * 	u(x) = Ax+b
   * \f}
   * for constant matrices \f$A\in\mathbb{R}^{3,3}\f$ and
   * \f$b\in\mathbb{R}^{3}\f$. The coefficient components are read in from a
   * parameter file.
   */
  class ExactSolutionLin_Data
  {
  public:
    /*!
     * Constructor
     */
    ExactSolutionLin_Data(const std::string &parameter_filename);

    /*!
     * Declare all parameters for tensor valued diffusion.
     *
     * @param prm
     */
    static void
      declare_parameters(ParameterHandler &prm);

    /*!
     * Parse all delaced parameters in file.
     *
     * @param prm
     */
    void
      parse_parameters(ParameterHandler &prm);

    /*!
     * Matrix coefficient \f$A\f$ of \f$u=Ax+b\f$.
     */
    Tensor<2, 3> A;

    /*!
     * Vector coefficient \f$b\f$ of \f$u=Ax+b\f$.
     */
    Tensor<1, 3> b;

    /*!
     * Divergence of u is simply the trace of \f$A\f$.
     */
    double div_u;

    /*!
     * The curl of \f$u\f$ if simply the representation vector
     * of the anti-symmetric part of the linear map represented by \f$A\f$.
     */
    Tensor<1, 3> curl_u;
  };

  /*!
   * @class ExactSolutionLin
   *
   * @brief Class for analytic solution \f$u=Ax+b\f$
   *
   * This class contains the actual implementation of an analytical solution
   * read from a parameter file if \f$u\f$ is a vector proxy, i.e., a 1-form or
   * a 2-form. The analytic solution is \f{eqnarray}{ u(x) = Ax+b \f} for
   * constant matrices \f$A\in\mathbb{R}^{3,3}\f$ and \f$b\in\mathbb{R}^{3}\f$.
   * The coefficient components are read in from a parameter file.
   *
   * @note that if the coefficients of the equation are oscillatory so will be the right hand side. This class is hence mostly only
   * used for sanity checking (semantic debugging) of the implementation for
   * vector proxies.
   */
  class ExactSolutionLin : public TensorFunction<1, 3>,
                           public ExactSolutionLin_Data
  {
  public:
    /*!
     * Constructor
     *
     * @param parameter_filename
     */
    ExactSolutionLin(const std::string &parameter_filename);

    /*!
     * Tensor value function for \f$u=Ax+b\f$.
     *
     * @param p
     */
    Tensor<1, 3>
      value(const Point<3> &p) const override;

    /*!
     * Tensor value list function for \f$u=Ax+b\f$.
     *
     * @param[in] points
     * @param[out] values
     */
    void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<1, 3>> &  values) const;
  };

  /*!
   * @class ExactSolutionLin_B_div
   *
   * @brief Auxiliary variable \f$\sigma = -B\nabla\cdot
   * u\f$ for 1-forms
   *
   * Class defines function for the auxiliary variable \f$\sigma = -B\nabla\cdot
   * u\f$ for an \f$H^1\f$-\f$H(curl)\f$ problem, i.e., for 1-forms, from the
   * data class.
   */
  class ExactSolutionLin_B_div : public Function<3>,
                                 public ExactSolutionLin_Data
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     */
    ExactSolutionLin_B_div(const std::string &parameter_filename);

    /*!
     * Value function for \f$\sigma = -B\nabla\cdot u\f$.
     *
     * @param point
     * @param component
     */
    double
      value(const Point<3> &   point,
            const unsigned int component = 0) const override;

    /*!
     * Value list function for \f$\sigma = -B\nabla\cdot u\f$.
     *
     * @param[in] points
     * @param[out] values
     * @param component
     */
    void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        values,
                 const unsigned int           component = 0) const override;

  private:
    /*!
     * Scalar coefficient.
     */
    const Diffusion_B b;
  };

  /*!
   * @class ExactSolutionLin_A_curl
   *
   * @brief Auxiliary variable \f$\sigma = A\nabla\times
   * u\f$ for 2-forms
   *
   * Class defines function for the auxiliary variable \f$\sigma = A\nabla\times
   * u\f$ for an \f$H(\mathrm{curl})\f$-\f$H(\mathrm{div})\f$ problem, i.e., for
   * 2-forms, from the data class.
   */
  class ExactSolutionLin_A_curl : public TensorFunction<1, 3>,
                                  public ExactSolutionLin_Data
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     */
    ExactSolutionLin_A_curl(const std::string &parameter_filename);

    /*!
     * Value function for \f$\sigma = A\nabla\times u\f$.
     *
     * @param point
     * @return
     */
    virtual Tensor<1, 3>
      value(const Point<3> &point) const override;

    /*!
     * Value list function for \f$\sigma = A\nabla\times u\f$.
     *
     * @param[in] points
     * @param[out] values
     */
    void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<1, 3>> &  values) const override;

  private:
    /*!
     * Tensor coefficent.
     */
    const Diffusion_A a;
  };

} // end namespace EquationData

#endif /* INCLUDE_EQUATION_DATA_EQN_EXACT_SOLUTION_LIN_H_ */
