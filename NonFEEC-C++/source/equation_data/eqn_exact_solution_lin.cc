#include <equation_data/eqn_exact_solution_lin.h>

namespace EquationData
{
  using namespace dealii;

  ExactSolutionLin_Data::ExactSolutionLin_Data(
    const std::string &parameter_filename)
  {
    ParameterHandler prm;

    declare_parameters(prm);

    A.clear();
    b.clear();

    // open file
    std::ifstream parameter_file(parameter_filename);

    prm.parse_input(parameter_file,
                    /* filename = */ "generated_parameter.in",
                    /* last_line = */ "",
                    /* skip_undefined = */ true);
    parse_parameters(prm);

    div_u = A[0][0] + A[1][1] + A[2][2];

    curl_u[0] = A[2][1] - A[1][2];
    curl_u[1] = A[0][2] - A[2][0];
    curl_u[2] = A[1][0] - A[0][1];
  }

  void
    ExactSolutionLin_Data::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Equation parameters");
    {
      prm.enter_subsection("Exact solution");
      {
        prm.declare_entry("a_00",
                          "1",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_01",
                          "0",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_02",
                          "0",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_10",
                          "0",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_11",
                          "1",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_12",
                          "0",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_20",
                          "0",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_21",
                          "0",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");
        prm.declare_entry("a_22",
                          "1",
                          Patterns::Double(-100, 100),
                          "Matrix entry.");

        prm.declare_entry("b_0",
                          "1",
                          Patterns::Double(-100, 100),
                          "Vector entry.");
        prm.declare_entry("b_1",
                          "1",
                          Patterns::Double(-100, 100),
                          "Vector entry.");
        prm.declare_entry("b_2",
                          "1",
                          Patterns::Double(-100, 100),
                          "Vector entry.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void
    ExactSolutionLin_Data::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Equation parameters");
    {
      prm.enter_subsection("Exact solution");
      {
        for (unsigned int i = 0; i < 3; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                A[i][j] = prm.get_double("a_" + Utilities::int_to_string(i, 1) +
                                         Utilities::int_to_string(j, 1));
              } // end for ++j
            b[i] = prm.get_double("b_" + Utilities::int_to_string(i, 1));
          } // end for ++i
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  ExactSolutionLin::ExactSolutionLin(const std::string &parameter_filename)
    : TensorFunction<1, 3>()
    , ExactSolutionLin_Data(parameter_filename)
  {}

  Tensor<1, 3>
    ExactSolutionLin::value(const Point<3> &point) const
  {
    Tensor<1, 3> value = A * point + b;
    return value;
  }

  void
    ExactSolutionLin::value_list(const std::vector<Point<3>> &points,
                                 std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i].clear();

        values[i] = A * points[i] + b;
      }
  }

  ExactSolutionLin_B_div::ExactSolutionLin_B_div(
    const std::string &parameter_filename)
    : Function<3>(1)
    , ExactSolutionLin_Data(parameter_filename)
    , b(parameter_filename)
  {}

  double
    ExactSolutionLin_B_div::value(const Point<3> &point,
                                  const unsigned int /*component = 0*/) const
  {
    double value = b.value(point);

    value = -value * div_u;

    return value;
  }

  void
    ExactSolutionLin_B_div::value_list(
      const std::vector<Point<3>> &points,
      std::vector<double> &        values,
      const unsigned int /*component = 0*/) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    b.value_list(points, values);

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i] = -values[i] * div_u;
      }
  }

  ExactSolutionLin_A_curl::ExactSolutionLin_A_curl(
    const std::string &parameter_filename)
    : TensorFunction<1, 3>()
    , ExactSolutionLin_Data(parameter_filename)
    , a(parameter_filename)
  {}

  Tensor<1, 3>
    ExactSolutionLin_A_curl::value(const Point<3> &point) const
  {
    Tensor<1, 3> value = a.value(point) * curl_u;
    return value;
  }

  void
    ExactSolutionLin_A_curl::value_list(const std::vector<Point<3>> &points,
                                        std::vector<Tensor<1, 3>> &values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i] = a.value(points[i]) * curl_u;
      }
  }

} // end namespace EquationData
