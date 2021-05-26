#ifndef INCLUDE_BASIS_RAVIART_THOMAS_H_
#define INCLUDE_BASIS_RAVIART_THOMAS_H_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include <functions/my_mapping_q1.h>

namespace ShapeFun
{
  using namespace dealii;

  /*!
   * @class BasisRaviartThomas
   *
   * @brief Raviart-Thomas basis on given cell
   *
   * Class implements Raviart-Thomas basis functions for a given quadrilateral.
   *
   * @note The We need Piola transforms to guarantee \f$H(\mathrm{div})\f$ conformity.
   */
  template <int dim>
  class BasisRaviartThomas : public Function<dim>
  {
  public:
    BasisRaviartThomas() = delete;

    /*!
     * Constructor.
     *
     * @param cell
     * @param degree
     */
    BasisRaviartThomas(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      unsigned int                                             degree = 0);

    /*!
     * Copy constructor.
     *
     * @param basis
     */
    BasisRaviartThomas(BasisRaviartThomas<dim> &basis);

    /*!
     * Set the index of the basis function to be evaluated.
     *
     * @param index
     */
    void
      set_index(unsigned int index);

    /*!
     * Compute the value of a Raviart-Thomas function.
     *
     * @param p
     * @param value
     */
    virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;

    /*!
     * Compute the value of a Raviart-Thomas function for a list of points.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>> &  values) const override;

  private:
    /*!
     * The mapping used there is a Piola transform to guarantee conformity in
     * \f$H(\mathrm{div})\f$.
     */
    MyMappingQ1<dim> mapping;

    /*!
     * Raviart-Thomas element of a given order.
     */
    FE_RaviartThomas<dim> fe;

    /*!
     * Index of current basis function.
     */
    unsigned int index_basis;
  };

  // exernal template instantiations
  extern template class BasisRaviartThomas<2>;
  extern template class BasisRaviartThomas<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_RAVIART_THOMAS_H_ */
