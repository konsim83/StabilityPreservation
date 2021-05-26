#include <Q_Ned/q_ned_post_processor.h>

namespace QNed
{
  using namespace dealii;

  /**
   * Constructor
   */
  QNed_PostProcessor::QNed_PostProcessor(const std::string &parameter_filename,
                                         const bool         use_exact_solution,
                                         const std::string  exact)
    : a(parameter_filename)
    , b_inverse(parameter_filename, use_exact_solution)
    , exact(exact)
  {}

  std::vector<std::string>
    QNed_PostProcessor::get_names() const
  {
    std::vector<std::string> solution_names(1, exact + "div_u");
    solution_names.emplace_back(exact + "curl_u");
    solution_names.emplace_back(exact + "curl_u");
    solution_names.emplace_back(exact + "curl_u");
    solution_names.emplace_back(exact + "A_curl_u");
    solution_names.emplace_back(exact + "A_curl_u");
    solution_names.emplace_back(exact + "A_curl_u");

    return solution_names;
  }

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    QNed_PostProcessor::get_data_component_interpretation() const
  {
    // div u = -B_inv*sigma
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(1, DataComponentInterpretation::component_is_scalar);

    // curl u
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

    // A*curl u
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

    return interpretation;
  }

  UpdateFlags
    QNed_PostProcessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  void
    QNed_PostProcessor::evaluate_vector_field(
      const DataPostprocessorInputs::Vector<3> &inputs,
      std::vector<Vector<double>> &             computed_quantities) const
  {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());

    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());

    Assert(inputs.solution_values[0].size() == 4, ExcInternalError());

    std::vector<Tensor<2, 3>> a_values(n_quadrature_points);
    std::vector<double>       b_inverse_values(n_quadrature_points);

    // Evaluate A and B at quadrature points
    a.value_list(inputs.evaluation_points, a_values);
    b_inverse.value_list(inputs.evaluation_points, b_inverse_values);

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        // div u = -B*sigma
        computed_quantities[q](0) =
          -b_inverse_values[q] * inputs.solution_values[q][0];

        {
          // curl u
          Tensor<2, 3> grad_u;
          for (unsigned int d = 0; d < 3; ++d)
            grad_u[d] = inputs.solution_gradients[q][d + 1]; // assign row-wise

          // row index is function, column is derivative
          computed_quantities[q](1) =
            grad_u[2][1] - grad_u[1][2]; // d_2u_3 - d_3u_2
          computed_quantities[q](2) =
            grad_u[0][2] - grad_u[2][0]; // d_3u_1 - d_1u_3
          computed_quantities[q](3) =
            grad_u[1][0] - grad_u[0][1]; // d_1u_2 - d_2u_1
        }

        // A*curl u
        for (unsigned int d = 4; d < 7; ++d)
          {
            computed_quantities[q](d) = 0; // erase old stuff
            for (unsigned int i = 0; i < 3; ++i)
              computed_quantities[q](d) +=
                a_values[q][d - 4][i] * computed_quantities[q](i + 1);
          }
      }
  }

} // end namespace QNed
