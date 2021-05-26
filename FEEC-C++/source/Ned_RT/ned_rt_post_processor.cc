#include <Ned_RT/ned_rt_post_processor.h>

namespace NedRT
{
  using namespace dealii;

  /**
   * Constructor
   */
  NedRT_PostProcessor::NedRT_PostProcessor(
    const std::string &parameter_filename,
    const bool         use_exact_solution,
    const std::string  exact)
    : a_inverse(parameter_filename)
    , b(parameter_filename, use_exact_solution)
    , exact(exact)
  {}

  std::vector<std::string>
    NedRT_PostProcessor::get_names() const
  {
    std::vector<std::string> solution_names(3, exact + "curl_u");
    solution_names.emplace_back(exact + "div_u");
    solution_names.emplace_back(exact + "minus_B_div_u");

    return solution_names;
  }

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    NedRT_PostProcessor::get_data_component_interpretation() const
  {
    // curl u = A_inv*sigma
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(3,
                     DataComponentInterpretation::component_is_part_of_vector);

    // div u
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // B*div u
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }

  UpdateFlags
    NedRT_PostProcessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  void
    NedRT_PostProcessor::evaluate_vector_field(
      const DataPostprocessorInputs::Vector<3> &inputs,
      std::vector<Vector<double>> &             computed_quantities) const
  {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());

    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());

    Assert(inputs.solution_values[0].size() == 6, ExcInternalError());

    std::vector<Tensor<2, 3>> a_inverse_values(n_quadrature_points);
    std::vector<double>       b_values(n_quadrature_points);

    // Evaluate A and B at quadrature points
    a_inverse.value_list(inputs.evaluation_points, a_inverse_values);
    b.value_list(inputs.evaluation_points, b_values);

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        for (unsigned int d = 0; d < 3; ++d)
          {
            computed_quantities[q](d) = 0; // erase old stuff
            for (unsigned int i = 0; i < 3; ++i)
              computed_quantities[q](d) +=
                a_inverse_values[q][d][i] * inputs.solution_values[q][i];
          }

        // For the divergence first compute the gradient
        computed_quantities[q](3) = 0;
        for (unsigned int d = 3; d < 6; ++d)
          {
            computed_quantities[q](3) += inputs.solution_gradients[q][d][d - 3];
          }

        // Now multiply with B
        computed_quantities[q](4) = -b_values[q] * computed_quantities[q](3);
      }
  }

} // end namespace NedRT
