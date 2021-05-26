#ifndef INCLUDE_BASIS_NEDELEC_CURL_TPP_
#define INCLUDE_BASIS_NEDELEC_CURL_TPP_

#include <functions/basis_nedelec_curl.h>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  BasisNedelecCurl<dim>::BasisNedelecCurl(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    unsigned int                                             degree)
    : Function<dim>(dim)
    , mapping(cell)
    , fe(degree)
    , index_basis(0)
  {}


  template <int dim>
  BasisNedelecCurl<dim>::BasisNedelecCurl(BasisNedelecCurl<dim> &basis)
    : Function<dim>(dim)
    , mapping(basis.mapping)
    , fe(basis.fe)
    , index_basis(basis.index_basis)
  {}


  template <int dim>
  void
    BasisNedelecCurl<dim>::set_index(unsigned int index)
  {
    index_basis = index;
  }


  template <int dim>
  void
    BasisNedelecCurl<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  value) const
  {
    Point<dim> p_ref = mapping.map_real_to_unit_cell(p);

    // This is the inverse of the jacobian \hat K -> K
    FullMatrix<double> jacobian = mapping.jacobian_map_unit_cell_to_real(p_ref);

    Tensor<2, dim> gradient;
    Vector<double> tmp;
    for (unsigned int d = 0; d < dim; ++d)
      {
        // Filling tensor row-wise
        gradient[d] = fe.shape_grad_component(index_basis, p_ref, d);
      }

    Vector<double> shape_curl(dim);
    shape_curl(0) = gradient[2][1] - gradient[1][2];
    shape_curl(1) = gradient[0][2] - gradient[2][0];
    shape_curl(2) = gradient[1][0] - gradient[0][1];

    jacobian.vmult(value, shape_curl);
    value /= jacobian.determinant();
  }


  template <int dim>
  void
    BasisNedelecCurl<dim>::vector_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    std::vector<Point<dim>> points_ref(points.size());
    mapping.map_real_to_unit_cell(points, points_ref);

    std::vector<FullMatrix<double>> jacobians(points.size(),
                                              FullMatrix<double>(dim, dim));
    mapping.jacobian_map_unit_cell_to_real(points_ref, jacobians);

    Tensor<2, dim> gradient;
    Vector<double> shape_curl(dim);
    for (unsigned int p = 0; p < points.size(); ++p)
      {
        shape_curl = 0;
        gradient.clear();
        for (unsigned int d = 0; d < dim; ++d)
          {
            // Filling tensor row-wise
            gradient[d] =
              fe.shape_grad_component(index_basis, points_ref[p], d);
          }

        shape_curl(0) = gradient[2][1] - gradient[1][2];
        shape_curl(1) = gradient[0][2] - gradient[2][0];
        shape_curl(2) = gradient[1][0] - gradient[0][1];

        jacobians[p].vmult(values[p], shape_curl);
        values[p] /= jacobians[p].determinant();

      } // end ++p
  }


  template <int dim>
  void
    BasisNedelecCurl<dim>::tensor_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<Tensor<1, dim>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    std::vector<Point<dim>> points_ref(points.size());
    mapping.map_real_to_unit_cell(points, points_ref);

    std::vector<FullMatrix<double>> jacobians(points.size(),
                                              FullMatrix<double>(dim, dim));
    mapping.jacobian_map_unit_cell_to_real(points_ref, jacobians);

    Tensor<2, dim> gradient;
    Vector<double> shape_curl(dim), value_tmp(dim);
    for (unsigned int p = 0; p < points.size(); ++p)
      {
        shape_curl = 0;
        value_tmp  = 0;
        gradient.clear();
        for (unsigned int d = 0; d < dim; ++d)
          {
            // Filling tensor row-wise
            gradient[d] =
              fe.shape_grad_component(index_basis, points_ref[p], d);
          }

        shape_curl(0) = gradient[2][1] - gradient[1][2];
        shape_curl(1) = gradient[0][2] - gradient[2][0];
        shape_curl(2) = gradient[1][0] - gradient[0][1];

        jacobians[p].vmult(value_tmp, shape_curl);
        value_tmp /= jacobians[p].determinant();

        for (unsigned int d = 0; d < dim; ++d)
          values[p][d] = value_tmp(d);

      } // end ++p
  }

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_NEDELEC_CURL_TPP_ */
