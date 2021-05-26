#ifndef EQN_COEFF_R_H_
#define EQN_COEFF_R_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

// std library
#include <cmath>
#include <cstdlib>
#include <vector>

namespace EquationData
{
  using namespace dealii;

  /*!
   * @class ReactionRate
   *
   * @brief Class for zero order terms. Handle with care! Not used by default.
   *
   * This zero order term that can possibly regularize the weak form if the
   * k-form is not a vector proxy. If it vanishes we have a Darcy problem for
   * \f$k=0\f$ or \f$k=3\f$.
   *
   * @note Handle non-zero values with care since then compatibility of elements is in general not guaranteed!
   */
  class ReactionRate : public Function<3>
  {
  public:
    /*!
     * Constructor.
     */
    ReactionRate()
      : Function<3>()
    {}

    /*!
     * Implementation of zero order term.
     *
     * @param points
     * @param values
     * @param component = 0
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        values,
                 const unsigned int           component = 0) const override;
  };

} // end namespace EquationData

#endif /* EQN_COEFF_R_H_ */
