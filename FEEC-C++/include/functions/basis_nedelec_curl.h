#ifndef INCLUDE_BASIS_NEDELEC_CURL_H_
#define INCLUDE_BASIS_NEDELEC_CURL_H_

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
   * @class BasisNedelecCurl
   *
   * @brief Curl of Nedelec basis on a given cell
   *
   * Class implements curls of vectorial Nedelec basis for a given
   * quadrilateral.
   *
   * @note The curls of \f$H(\mathrm{curl})\f$ conforming functions are in \f$H(\mathrm{div})\f$.
   * So we need Piola transforms to guarantee \f$H(\mathrm{div})\f$ conformity.
   */
  template <int dim>
  class BasisNedelecCurl : public Function<dim>
  {
  public:
    BasisNedelecCurl() = delete;

    /*!
     * Constructor.
     *
     * @param cell
     * @param degree
     */
    BasisNedelecCurl(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      unsigned int                                             degree = 0);

    /*!
     * Copy constructor.
     *
     * @param basis
     */
    BasisNedelecCurl(BasisNedelecCurl<dim> &basis);

    /*!
     * Set the index of the basis function to be evaluated.
     *
     * @param index
     */
    void
      set_index(unsigned int index);

    /*!
     * Compute the curl of a Nedelec function.
     *
     * @param p
     * @param value
     */
    virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;

    /*!
     * Compute the curl of a Nedelec function for a list of points.
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>> &  values) const override;

    /*!
     * Compute the curl of a Nedelec function for a list of points. Return
     * tensors.
     *
     * @param[in] points
     * @param[out] values
     */
    void
      tensor_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Tensor<1, dim>> &  values) const;

  private:
    /*!
     * The mapping used there is a Piola transform to guarantee conformity in
     * \f$H(\mathrm{div})\f$.
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
  extern template class BasisNedelecCurl<2>;
  extern template class BasisNedelecCurl<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_NEDELEC_CURL_H_ */
