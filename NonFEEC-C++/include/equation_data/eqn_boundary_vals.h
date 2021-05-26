#ifndef EQN_BOUNDARY_VALS_H_
#define EQN_BOUNDARY_VALS_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

// std library
#include <cmath>
#include <cstdlib>
#include <vector>

/*!
 * @namespace EquationData
 *
 * @brief Holds all function objects and their data objects.
 */
namespace EquationData
{
  using namespace dealii;

  /*!
   * @class BoundaryValues_u
   *
   * @brief Scalar boundary values for \f$u\f$ which are essential for 0-forms and natural for 3-forms.
   */
  class BoundaryValues_u : public Function<3>
  {
  public:
    /*!
     * Constructor.
     */
    BoundaryValues_u()
      : Function<3>(1)
    {}

    /*!
     * Implementation of scalar boundary values for \f$u\f$.
     *
     * @param[in] points
     * @param[out] values
     * @param[in] component
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        values,
                 const unsigned int           component = 0) const override;
  };

  /*!
   * @class Boundary_B_div_u
   *
   * @brief Boundary values for \f$B \nabla\cdot u\f$ which are essential for 1-forms and natural for 2-forms.
   */
  class Boundary_B_div_u : public Function<3>
  {
  public:
    /*!
     * Constructor.
     */
    Boundary_B_div_u()
      : Function<3>(1)
    {}

    /*!
     * Implementation of boundary values for \f$B \nabla\cdot u\f$.
     *
     * @param[in] points
     * @param[out] values
     * @param component
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        values,
                 const unsigned int           component = 0) const override;
  };

  /*!
   * @class Boundary_A_curl_u
   *
   * @brief Boundary values for \f$A \nabla\times u\f$ which are essential for 2-forms and natural for 1-forms.
   */
  class Boundary_A_curl_u : public TensorFunction<1, 3>
  {
  public:
    /*!
     * Constructor.
     */
    Boundary_A_curl_u()
      : TensorFunction<1, 3>()
    {}

    /*!
     * Implementation of boundary values for \f$A\nabla \times u\f$.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<1, 3>> &  values) const override;
  };

  /*!
   * @class Boundary_A_grad_u
   *
   * @brief Boundary values for \f$A\nabla u\f$ which are essential for 3-forms and
   * natural for 0-forms.
   */
  class Boundary_A_grad_u : public TensorFunction<1, 3>
  {
  public:
    /*!
     * Constructor.
     */
    Boundary_A_grad_u()
      : TensorFunction<1, 3>()
    {}

    /*!
     * Implementation of boundary values for \f$A\nabla u\f$.
     *
     * @param point
     */
    virtual Tensor<1, 3>
      value(const Point<3> &point) const override;

    /*!
     * Implementation of boundary values for \f$A\nabla u\f$.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<1, 3>> &  values) const override;
  };

} // end namespace EquationData

#endif /* EQN_BOUNDARY_VALS_H_ */
