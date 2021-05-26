#include <RT_DQ/rt_dq_post_processor.h>

namespace RTDQ
{
  using namespace dealii;

  /**
   * Constructor
   */
  RTDQ_PostProcessor::RTDQ_PostProcessor(const std::string &parameter_filename)
    : a_inverse(parameter_filename)
  {}

  std::vector<std::string>
    RTDQ_PostProcessor::get_names() const
  {
    std::vector<std::string> solution_names(3, "grad_u");

    return solution_names;
  }

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    RTDQ_PostProcessor::get_data_component_interpretation() const
  {
    // grad u = -A_inverse*sigma
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(3,
                     DataComponentInterpretation::component_is_part_of_vector);

    return interpretation;
  }

  UpdateFlags
    RTDQ_PostProcessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  void
    RTDQ_PostProcessor::evaluate_vector_field(
      const DataPostprocessorInputs::Vector<3> &inputs,
      std::vector<Vector<double>> &             computed_quantities) const
  {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());

    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());

    Assert(inputs.solution_values[0].size() == 4, ExcInternalError());

    std::vector<Tensor<2, 3>> a_inverse_values(n_quadrature_points);

    // Evaluate A and B at quadrature points
    a_inverse.value_list(inputs.evaluation_points, a_inverse_values);

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        // gradients
        for (unsigned int d = 0; d < 3; ++d)
          {
            computed_quantities[q](d) = 0; // erase old stuff
            for (unsigned int i = 0; i < 3; ++i)
              computed_quantities[q](d) -=
                a_inverse_values[q][d][i] * inputs.solution_values[q](i);
          }
      }
  }

} // end namespace RTDQ
