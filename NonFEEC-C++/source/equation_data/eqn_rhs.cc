#include <equation_data/eqn_rhs.h>

namespace EquationData
{
  using namespace dealii;

  RightHandSideExactLin::RightHandSideExactLin(
    const std::string &parameter_filename)
    : RightHandSide(3)
    , ExactSolutionLin_Data(parameter_filename)
    , Diffusion_A_Data(parameter_filename)
    , Diffusion_B_Data(parameter_filename)
  {
    R_trans_curl_u = transpose(rot) * curl_u;
  }

  void
    RightHandSideExactLin::tensor_value_list(
      const std::vector<Point<3>> &points,
      std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i].clear();

        // -grad (B div_u), works only with a certain
        values[i][0] =
          2 * pi * k * div_u * scale * alpha * cos(2 * pi * k * points[i](0));
        values[i][1] = 0;
        values[i][2] = 0;

        values[i][0] +=
          2 * pi *
          (-rot[2][1] * scale_y * alpha_y * k_y *
             cos(2 * pi * k_y * points[i](1)) * R_trans_curl_u[1] +
           rot[1][2] * scale_z * alpha_z * k_z *
             cos(2 * pi * k_z * points[i](2)) * R_trans_curl_u[2]);

        values[i][1] +=
          2 * pi *
          (-rot[0][2] * scale_z * alpha_z * k_z *
             cos(2 * pi * k_z * points[i](2)) * R_trans_curl_u[2] +
           rot[2][0] * scale_x * alpha_x * k_x *
             cos(2 * pi * k_x * points[i](0)) * R_trans_curl_u[0]);

        values[i][2] +=
          2 * pi *
          (-rot[1][0] * scale_x * alpha_x * k_x *
             cos(2 * pi * k_x * points[i](0)) * R_trans_curl_u[0] +
           rot[0][1] * scale_y * alpha_y * k_y *
             cos(2 * pi * k_y * points[i](1)) * R_trans_curl_u[1]);
      }
  }

  RightHandSideParsed::RightHandSideParsed(
    const std::string &parameter_filename,
    unsigned int       n_components)
    : RightHandSide(n_components)
  {
    // A parameter handler
    ParameterHandler prm;

    // Declare a section for the function we need
    prm.enter_subsection("Equation parameters");
    prm.enter_subsection("Right-hand side");
    Functions::ParsedFunction<3>::declare_parameters(prm, n_components);
    prm.leave_subsection();
    prm.leave_subsection();

    // open file
    std::ifstream parameter_file(parameter_filename);

    // Parse an input file.
    prm.parse_input(parameter_file,
                    /* filename = */ "generated_parameter.in",
                    /* last_line = */ "",
                    /* skip_undefined = */ true);

    // Initialize the ParsedFunction object with the given file
    prm.enter_subsection("Equation parameters");
    prm.enter_subsection("Right-hand side");
    this->parse_parameters(prm);
    prm.leave_subsection();
    prm.leave_subsection();
  }

  void
    RightHandSideParsed::tensor_value_list(
      const std::vector<Point<3>> &points,
      std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i].clear();

        for (unsigned int d = 0; d < 3; ++d)
          {
            values[i][d] = value(points[i], /* component */ d);
          }
      }
  }

} // end namespace EquationData
