#ifndef INCLUDE_BASIS_RAVIART_THOMAS_TPP_
#define INCLUDE_BASIS_RAVIART_THOMAS_TPP_

#include <functions/basis_raviart_thomas.h>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  BasisRaviartThomas<dim>::BasisRaviartThomas(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    unsigned int                                             degree)
    : Function<dim>(dim)
    , mapping(cell)
    , fe(degree)
    , index_basis(0)
  {}


  template <int dim>
  BasisRaviartThomas<dim>::BasisRaviartThomas(BasisRaviartThomas<dim> &basis)
    : Function<dim>(dim)
    , mapping(basis.mapping)
    , fe(basis.fe)
    , index_basis(basis.index_basis)
  {}


  template <int dim>
  void
    BasisRaviartThomas<dim>::set_index(unsigned int index)
  {
    index_basis = index;
  }


  template <int dim>
  void
    BasisRaviartThomas<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &  value) const
  {
    Point<dim> p_ref = mapping.map_real_to_unit_cell(p);

    // This is the inverse of the jacobian \hat K -> K
    FullMatrix<double> jacobians =
      mapping.jacobian_map_unit_cell_to_real(p_ref);

    Vector<double> tmp(dim);
    for (unsigned int d = 0; d < dim; ++d)
      {
        tmp(d) = fe.shape_value_component(index_basis, p_ref, d);
      }

    jacobians.vmult(value, tmp);
    value /= jacobians.determinant();
  }


  template <int dim>
  void
    BasisRaviartThomas<dim>::vector_value_list(
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

    Vector<double> tmp(dim);

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        tmp = 0;
        for (unsigned int d = 0; d < dim; ++d)
          {
            tmp(d) = fe.shape_value_component(index_basis, points_ref[p], d);
          }
        jacobians[p].vmult(values[p], tmp);
        values[p] /= jacobians[p].determinant();
      } // end ++p
  }

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_RAVIART_THOMAS_TPP_ */
