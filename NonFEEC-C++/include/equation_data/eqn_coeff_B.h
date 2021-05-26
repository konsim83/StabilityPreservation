#ifndef EQN_COEFF_B_H_
#define EQN_COEFF_B_H_

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

// std library
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace EquationData
{
  using namespace dealii;

  /*!
   * @class Diffusion_B_Data
   *
   * @brief Class containing data to construct a positive scalar.
   *
   * Class containing data for a scalar valued coefficient that only occurs when
   * 1-forms or 2-forms are considered.
   */
  class Diffusion_B_Data
  {
  public:
    /*!
     * Constructor
     */
    Diffusion_B_Data(const std::string &parameter_filename);

    static void
      declare_parameters(ParameterHandler &prm);
    void
      parse_parameters(ParameterHandler &prm);

    /*!
     * Frequency of oscillations
     */
    unsigned int k;

    /*!
     * Scaling factor
     */
    double scale;

    /*!
     * Scaling factor for oscillations
     */
    double alpha;

    /*!
     * Function expression in muParser format.
     */
    std::string expression;
  };

  /*!
   * @class Diffusion_B
   *
   * @brief Class containing a positive scalar coefficient.
   *
   * Second (scalar) coefficient function. Must be positive definite and
   * uniformly bounded from below and above.
   *
   * @note The expression of the coefficient is only used if the sanity check using analytic solutions is set to false.
   * In case the user wishes to use an analytic (manufactured) solution the
   *coeffcient is given as \f{eqnarray}{ B_\varepsilon(x,y,z) =
   *\mathrm{scale}*(1-\mathrm{alpha}*\sin(2\pi * \mathrm{frequency}* x)) \f}
   * where \f$\mathrm{scale}, \mathrm{alpha}, \mathrm{frequency}\f$ are the
   *constants provided by the user in the parameter file.
   */
  class Diffusion_B : public FunctionParser<3>, public Diffusion_B_Data
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     * @param use_exact_solution
     */
    Diffusion_B(const std::string &parameter_filename,
                bool               use_exact_solution = false);

  private:
    /*!
     * Declaration of parsed variables.
     */
    const std::string variables = "x,y,z";

    /*!
     * Contains the parsed function expression.
     */
    std::string fnc_expression;

    /*!
     * Constais a list of parsed user-defined constants.
     */
    std::map<std::string, double> constants;
  };

  /*!
   * @class DiffusionInverse_B
   *
   * @brief Class containing the **inverse** of a positive scalar coefficient.
   *
   * Inverse of second scalar valued coefficient. Must be positive definite and
   * uniformly bounded from below and above.
   *
   * @note The expression of the coefficient is only used if the sanity check using analytic solutions is set to false.
   * In case the user wishes to use an analytic (manufactured) solution the
   *coeffcient is given as \f{eqnarray}{ B_\varepsilon(x) =
   *\mathrm{scale}*(1-\mathrm{alpha}*\sin(2\pi * \mathrm{frequency}* x_1)) \f}
   * where \f$\mathrm{scale}, \mathrm{alpha}, \mathrm{frequency}\f$ are the
   *constants provided by the user in the parameter file.
   */
  class DiffusionInverse_B : public FunctionParser<3>, public Diffusion_B_Data
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     * @param use_exact_solution
     */
    DiffusionInverse_B(const std::string &parameter_filename,
                       bool               use_exact_solution = false);

  private:
    /*!
     * Declaration of parsed variables.
     */
    const std::string variables = "x,y,z";

    /*!
     * Contains the parsed function expression.
     */
    std::string inverse_fnc_expression;

    /*!
     * Constais a list of parsed user-defined constants.
     */
    std::map<std::string, double> constants;
  };

} // end namespace EquationData

#endif /* EQN_COEFF_B_H_ */
