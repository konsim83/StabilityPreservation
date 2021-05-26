#include <Q/q_post_processor.h>

namespace Q
{
  using namespace dealii;

  /**
   * Constructor
   */
  Q_PostProcessor::Q_PostProcessor(const std::string &parameter_filename)
    : a(parameter_filename)
  {}

  std::vector<std::string>
    Q_PostProcessor::get_names() const
  {
    std::vector<std::string> solution_names(3, "grad_u");
    solution_names.emplace_back("minus_A_grad_u");
    solution_names.emplace_back("minus_A_grad_u");
    solution_names.emplace_back("minus_A_grad_u");

    return solution_names;
  }

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    Q_PostProcessor::get_data_component_interpretation() const
  {
    // grad u
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(3,
                     DataComponentInterpretation::component_is_part_of_vector);

    // -A grad u
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

    return interpretation;
  }

  UpdateFlags
    Q_PostProcessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  void
    Q_PostProcessor::evaluate_scalar_field(
      const DataPostprocessorInputs::Scalar<3> &inputs,
      std::vector<Vector<double>> &             computed_quantities) const
  {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());

    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());

    std::vector<Tensor<2, 3>> a_values(n_quadrature_points);

    // Evaluate A and B at quadrature points
    a.value_list(inputs.evaluation_points, a_values);

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        // gradients
        for (unsigned int d = 0; d < 3; ++d)
          computed_quantities[q](d) =
            inputs.solution_gradients[q][d]; // assign row-wise

        // -A*grad u
        for (unsigned int d = 3; d < 6; ++d)
          {
            computed_quantities[q](d) = 0; // erase old stuff
            for (unsigned int i = 0; i < 3; ++i)
              computed_quantities[q](d) -=
                a_values[q][d - 3][i] * computed_quantities[q](i);
          }
      }
  }

} // end namespace Q
