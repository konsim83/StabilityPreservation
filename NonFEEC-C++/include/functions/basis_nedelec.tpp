#ifndef INCLUDE_BASIS_NEDELEC_TPP_
#define INCLUDE_BASIS_NEDELEC_TPP_

#include <functions/basis_nedelec.h>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  BasisNedelec<dim>::BasisNedelec(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    unsigned int                                             degree)
    : Function<dim>(dim)
    , mapping(cell)
    , fe(degree)
    , index_basis(0)
  {}


  template <int dim>
  BasisNedelec<dim>::BasisNedelec(BasisNedelec<dim> &basis)
    : Function<dim>(dim)
    , mapping(basis.mapping)
    , fe(basis.fe)
    , index_basis(basis.index_basis)
  {}


  template <int dim>
  void
    BasisNedelec<dim>::set_index(unsigned int index)
  {
    index_basis = index;
  }


  template <int dim>
  void
    BasisNedelec<dim>::vector_value(const Point<dim> &p,
                                    Vector<double> &  value) const
  {
    Point<dim> p_ref = mapping.map_real_to_unit_cell(p);

    // This is the inverse of the jacobian \hat K -> K
    FullMatrix<double> inv_jacobian = mapping.jacobian_map_real_to_unit_cell(p);

    Vector<double> tmp(dim);
    for (unsigned int d = 0; d < dim; ++d)
      {
        tmp(d) = fe.shape_value_component(index_basis, p_ref, d);
      }

    inv_jacobian.Tvmult(value, tmp);
  }


  template <int dim>
  void
    BasisNedelec<dim>::vector_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    std::vector<Point<dim>> points_ref(points.size());
    mapping.map_real_to_unit_cell(points, points_ref);

    std::vector<FullMatrix<double>> inv_jacobians(points.size(),
                                                  FullMatrix<double>(dim, dim));
    mapping.jacobian_map_real_to_unit_cell(points, inv_jacobians);

    Vector<double> tmp(dim);

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        tmp = 0;
        for (unsigned int d = 0; d < dim; ++d)
          {
            tmp(d) = fe.shape_value_component(index_basis, points_ref[p], d);
          }
        inv_jacobians[p].Tvmult(values[p], tmp);
      } // end ++p
  }

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_NEDELEC_TPP_ */
