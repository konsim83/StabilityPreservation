#ifndef EQN_COEFF_A_H_
#define EQN_COEFF_A_H_

#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

// std library
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace EquationData
{
  using namespace dealii;

  /*!
   * @class Diffusion_A_Data
   *
   * @brief Class containing data to construct a positive tensor.
   *
   * This class implements a data object that serves as a base class for
   * concrete implementations. It contains information about anisotropy and
   * inhomogeneity.
   */
  class Diffusion_A_Data
  {
  public:
    /*!
     * Constructor
     *
     * @param parameter_filename
     */
    Diffusion_A_Data(const std::string &parameter_filename);

    static void
      /*!
       * Declare all paramters to be used.
       *
       * @param prm
       */
      declare_parameters(ParameterHandler &prm);
    void
      /*!
       * Parse all paramters to be used.
       *
       * @param prm
       */
      parse_parameters(ParameterHandler &prm);

    /*!
     * Frequency of oscillations in x, y and z.
     */
    unsigned int k_x, k_y, k_z;

    /*!
     * Scaling factor in x, y and z..
     */
    double scale_x, scale_y, scale_z;

    /*!
     * Scaling factor for oscillations in x, y and z.
     */
    double alpha_x, alpha_y, alpha_z;

    /*!
     * Three Euler angles.
     */
    const double alpha_, beta_, gamma_;

    /*!
     * True if tensor coefficient should be rotated in space.
     */
    bool rotate;

    /*!
     * Description of rotation with Euler angles. This rotates the
     * tensor coefficients in space and allows for the construction
     * of more general symmetric positive definite data. If 'rotate=false'
     * then 'rot' is just the identity.
     */
    Tensor<2, 3> rot;
  };

  /*!
   * @class Diffusion_A
   *
   * @brief Class containing a positive tensor.
   *
   * This class implements a uniformly positive tensor that can be interpreted
   * differently depending on what object (k-form) they act on. If the object is
   * a 0-form or a 3-form then it can be interpreted as a diffusivity. If the
   * object it acts on is a vector proxy, i.e., a 1-form or a 2-form, it can
   * represent permittivity and permeability tensors or magneto-electric
   * tensors.
   *
   * The current implementation can be strongly anisotropic and inhomogeneous
   *and is in our implementation given by \f{eqnarray}{ A_\varepsilon(x,y,z) = R
   *	\left(
   *	\begin{array}{ccc}
   *		\mathrm{scale}_x*(1-\mathrm{alpha}_x*\sin(2\pi * \mathrm{frequency}_x*
   *x)) & 0 & 0 \\
   *		0 & \mathrm{scale}_y*(1-\mathrm{alpha}_y*\sin(2\pi *
   *\mathrm{frequency}_y* y)) & 0 \\
   *		0 & 0 & \mathrm{scale}_z*(1-\mathrm{alpha}_z*\sin(2\pi *
   *\mathrm{frequency}_z* z)) \\ \end{array} \right) R^T \f} where
   *\f$\mathrm{scale}_x, \mathrm{scale}_y, \mathrm{scale}_z, \mathrm{alpha}_x,
   *\mathrm{alpha}_y, \mathrm{alpha}_z, \mathrm{frequency}_x,
   *\mathrm{frequency}_y, \mathrm{frequency}_z\f$ are the constants provided by
   *the user in the parameter file.
   */
  class Diffusion_A : public TensorFunction<2, 3>, public Diffusion_A_Data
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     */
    Diffusion_A(const std::string &parameter_filename)
      : TensorFunction<2, 3>()
      , Diffusion_A_Data(parameter_filename)
    {}

    /*!
     * Implementation of the tensor.
     * Must be positive definite and uniformly bounded.
     */
    virtual Tensor<2, 3>
      value(const Point<3> &point) const override;

    /*!
     * Implementation of the tensor.
     * Must be positive definite and uniformly bounded.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<2, 3>> &  values) const override;
  };

  /*!
   * @class DiffusionInverse_A
   *
   * @brief Class containing the inverse of a positive tensor.
   *
   * Same as Diffusion_A but represents the inverse tensor.
   *
   * The current implementation can be strongly anisotropic and inhomogeneous
   *and is in our implementation given by \f{eqnarray}{ A_\varepsilon(x,y,z) = R
   *	\left(
   *	\begin{array}{ccc}
   *		\mathrm{scale}_x*(1-\mathrm{alpha}_x*\sin(2\pi * \mathrm{frequency}_x*
   *x)) & 0 & 0 \\
   *		0 & \mathrm{scale}_y*(1-\mathrm{alpha}_y*\sin(2\pi *
   *\mathrm{frequency}_y* y)) & 0 \\
   *		0 & 0 & \mathrm{scale}_z*(1-\mathrm{alpha}_z*\sin(2\pi *
   *\mathrm{frequency}_z* z)) \\ \end{array} \right) R^T \f} where
   *\f$\mathrm{scale}_x, \mathrm{scale}_y, \mathrm{scale}_z, \mathrm{alpha}_x,
   *\mathrm{alpha}_y, \mathrm{alpha}_z, \mathrm{frequency}_x,
   *\mathrm{frequency}_y, \mathrm{frequency}_z\f$ are the constants provided by
   *the user in the parameter file.
   */
  class DiffusionInverse_A : public TensorFunction<2, 3>,
                             public Diffusion_A_Data
  {
  public:
    /*!
     * Constructor.
     *
     * @param parameter_filename
     */
    DiffusionInverse_A(const std::string &parameter_filename)
      : TensorFunction<2, 3>()
      , Diffusion_A_Data(parameter_filename)
    {}

    /*!
     * Implementation of inverse of the tensor.
     * Must be positive definite and uniformly bounded.
     */
    virtual Tensor<2, 3>
      value(const Point<3> &point) const override;

    /*!
     * Implementation of inverse of the tensor.
     * Must be positive definite and uniformly bounded.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<2, 3>> &  values) const override;
  };

} // end namespace EquationData

#endif /* EQN_COEFF_A_H_ */
