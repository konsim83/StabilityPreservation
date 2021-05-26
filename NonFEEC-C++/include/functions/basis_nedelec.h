#ifndef INCLUDE_BASIS_NEDELEC_H_
#define INCLUDE_BASIS_NEDELEC_H_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_nedelec.h>
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
   * @class BasisNedelec
   *
   * @brief Nedelec basis on a given cell
   *
   * Class implements values of Nedelec basis functions for a given
   * quadrilateral.
   *
   * @note The We need covariant transforms to guarantee \f$H(\mathrm{curl})\f$ conformity.
   */
  template <int dim>
  class BasisNedelec : public Function<dim>
  {
  public:
    BasisNedelec() = delete;

    /*!
     * Constructor.
     *
     * @param cell
     * @param degree
     */
    BasisNedelec(const typename Triangulation<dim>::active_cell_iterator &cell,
                 unsigned int degree = 0);

    /*!
     * Copy constructor.
     *
     * @param basis
     */
    BasisNedelec(BasisNedelec<dim> &basis);

    /*!
     * Set the index of the basis function to be evaluated.
     *
     * @param index
     */
    void
      set_index(unsigned int index);

    /*!
     * Compute the value of a Nedelec function.
     *
     * @param p
     * @param value
     */
    virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;

    /*!
     * Compute the value of a Nedelec function for a list of points.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>> &  values) const override;

  private:
    /*!
     * The mapping used there is a covariant transform to guarantee conformity
     * in \f$H(\mathrm{curl})\f$.
     */
    MyMappingQ1<dim> mapping;

    /*!
     * Nedelec element of a given order.
     */
    FE_Nedelec<dim> fe;

    /*!
     * Index of current basis function.
     */
    unsigned int index_basis;
  };

  // exernal template instantiations
  extern template class BasisNedelec<2>;
  extern template class BasisNedelec<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_NEDELEC_H_ */
