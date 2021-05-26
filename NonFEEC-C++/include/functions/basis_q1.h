#ifndef INCLUDE_BASIS_Q1_H_
#define INCLUDE_BASIS_Q1_H_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/full_matrix.h>

// STL
#include <cmath>
#include <fstream>

// My Headers

/*!
 * @namespace ShapeFun
 *
 * @brief contains tools to evaluate standard finite element shape functions on arbitrary cells in
 * various spaces that are approximated conformally.
 */
namespace ShapeFun
{
  using namespace dealii;

  /*!
   * @class BasisQ1
   *
   * @brief \f$Q_1\f$ basis on given cell
   *
   * Class implements scalar \f$Q_1\f$-basis functions for a given
   * quadrilateral.
   */
  template <int dim>
  class BasisQ1 : public Function<dim>
  {
  public:
    BasisQ1() = delete;

    /*!
     * Constructor. Template specialization \f$dim=2\f$.
     * @param cell
     *
     * For dim=2 build up coefficient matrix \f$A=(a_{ij})\f$ for basis
     * polynomial \f$\varphi_i(x,y)=a_i^0 + a_i^1 x + a_i^2 y + a_i^3 xy \f$.
     * For dim=3 build up coefficient matrix \f$A=(a_{ij})\f$ for basis
     * polynomial \f$\varphi_i(x,y)=a_i^0 + a_i^1 x + a_i^2 y + a_i^3 xy \f$.
     * The \f$i\f$-th column of the matrix hence contains the coefficients for
     * the \f$i\f$-th basis associated to the \f$i\f$-th vertex.
     */
    BasisQ1(const typename Triangulation<dim>::active_cell_iterator &cell);

    /*!
     * Copy constructor.
     */
    BasisQ1(const BasisQ1<dim> &);

    /*!
     * Set the index of the basis function to be evaluated.
     *
     * @param index
     */
    void
      set_index(unsigned int index);

    /*!
     * Evaluate a basis function with a preset index at one given point in 2D or
     * 3D.
     *
     * @param p
     * @param component
     */
    virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

    /*!
     * Evaluate a basis function with a preset index at given point list in 2D
     * and 3D.
     *
     * @param[in] points
     * @param[out] values
     * @param component
     */
    virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double> &          values,
                 const unsigned int             component = 0) const override;

  private:
    /*!
     * Index of current basis function to be evaluated.
     */
    unsigned int index_basis;

    /*!
     * Matrix columns hold coefficients of basis functions.
     */
    FullMatrix<double> coeff_matrix;
  };

  /*
   * Declare specializations in 2D
   */
  template <>
  BasisQ1<2>::BasisQ1(
    const typename Triangulation<2>::active_cell_iterator &cell);

  template <>
  double
    BasisQ1<2>::value(const Point<2> &, const unsigned int) const;

  template <>
  void
    BasisQ1<2>::value_list(const std::vector<Point<2>> &,
                           std::vector<double> &,
                           const unsigned int) const;

  /*
   * Declare specializations in 3D
   */
  template <>
  double
    BasisQ1<3>::value(const Point<3> &, const unsigned int) const;

  template <>
  BasisQ1<3>::BasisQ1(
    const typename Triangulation<3>::active_cell_iterator &cell);

  template <>
  void
    BasisQ1<3>::value_list(const std::vector<Point<3>> &,
                           std::vector<double> &,
                           const unsigned int) const;

  // exernal template instantiations
  extern template class BasisQ1<2>;
  extern template class BasisQ1<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_Q1_H_ */
