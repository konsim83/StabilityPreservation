#include <equation_data/eqn_coeff_B.h>

namespace EquationData
{
  using namespace dealii;

  Diffusion_B_Data::Diffusion_B_Data(const std::string &parameter_filename)
  {
    ParameterHandler prm;

    declare_parameters(prm);

    // open file
    std::ifstream parameter_file(parameter_filename);

    prm.parse_input(parameter_file,
                    /* filename = */ "generated_parameter.in",
                    /* last_line = */ "",
                    /* skip_undefined = */ true);
    parse_parameters(prm);
  }

  void
    Diffusion_B_Data::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Equation parameters");
    {
      prm.enter_subsection("Diffusion B");
      {
        prm.declare_entry("frequency",
                          "0",
                          Patterns::Integer(0, 100),
                          "Frequency of coefficient.");
        prm.declare_entry("scale",
                          "1",
                          Patterns::Double(0.0001, 10000),
                          "Scaling parameter.");
        prm.declare_entry("alpha",
                          "1",
                          Patterns::Double(0.000, 10000),
                          "Scaling parameter for oscillatory part.");
        prm.declare_entry("Function expression",
                          "0",
                          Patterns::Anything(),
                          "Function expression as a string.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void
    Diffusion_B_Data::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Equation parameters");
    {
      prm.enter_subsection("Diffusion B");
      {
        k = prm.get_integer("frequency");

        scale = prm.get_double("scale");

        alpha = prm.get_double("alpha");

        expression = prm.get("Function expression");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  Diffusion_B::Diffusion_B(const std::string &parameter_filename,
                           bool               use_exact_solution)
    : FunctionParser<3>()
    , Diffusion_B_Data(parameter_filename)
    , fnc_expression(expression)
  {
    constants["pi"]        = numbers::PI;
    constants["frequency"] = k;
    constants["scale"]     = scale;
    constants["alpha"]     = alpha;

    /*
     * If we use an exact solution we need a specific function
     * expression and not the parsed one
     */
    if (use_exact_solution)
      fnc_expression = "scale * (1.0 - alpha * sin(2*pi*frequency*x))";

    this->initialize(variables, fnc_expression, constants);
  }

  DiffusionInverse_B::DiffusionInverse_B(const std::string &parameter_filename,
                                         bool               use_exact_solution)
    : FunctionParser<3>()
    , Diffusion_B_Data(parameter_filename)
  {
    /*
     * If we use an exact solution we need a specific function
     * expression and not the parsed one
     */
    if (use_exact_solution)
      inverse_fnc_expression =
        "1/(scale * (1.0 - alpha * sin(2*pi*frequency*x)))";
    else
      inverse_fnc_expression = "1/(" + expression + ")";

    constants["pi"]        = numbers::PI;
    constants["frequency"] = k;
    constants["scale"]     = scale;
    constants["alpha"]     = alpha;

    this->initialize(variables, inverse_fnc_expression, constants);
  }

} // end namespace EquationData
