#ifndef INCLUDE_Q_NED_POST_PROCESSOR_H_
#define INCLUDE_Q_NED_POST_PROCESSOR_H_

// deal.ii
#include <deal.II/numerics/data_postprocessor.h>
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>

#include <vector>

// my headers
#include <config.h>

namespace QNed
{
  using namespace dealii;

  /*!
   * @class QNed_PostProcessor
   *
   * @brief Class to postprocess a solution of a \f$H(\mathrm{grad})\f$-\f$H(\mathrm{curl})\f$ problem.
   *
   * This class computes quantities that are not computed by the solver because
   * they are either interesting or necessary for a comparison with other
   * solvers wich may use different geometric proxies.
   */
  class QNed_PostProcessor : public DataPostprocessor<3>
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     * @param use_exact_solution
     * @param exact
     */
    QNed_PostProcessor(const std::string &parameter_filename,
                       const bool         use_exact_solution,
                       const std::string  exact = "");

    /*!
     * This is the actual evaluation routine of the  post processor.
     */
    virtual void
      evaluate_vector_field(
        const DataPostprocessorInputs::Vector<3> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;

    /*!
     * Define all names of solution and post processed quantities.
     */
    virtual std::vector<std::string>
      get_names() const override;

    /*!
     * Define all interpretations of solution and post processed quantities.
     */
    virtual std::vector<
      DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

    /*!
     * Define all necessary update flags when looping over cells to be post
     * processed.
     */
    virtual UpdateFlags
      get_needed_update_flags() const override;

  private:
    const EquationData::Diffusion_A        a;
    const EquationData::DiffusionInverse_B b_inverse;

    std::string exact;
  };

} // end namespace QNed

#endif /* INCLUDE_Q_NED_POST_PROCESSOR_H_ */
