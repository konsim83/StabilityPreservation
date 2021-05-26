#ifndef INCLUDE_TRUNK_SCALAR_SHAPE_FUNCTION_GRAD_TPP_
#define INCLUDE_TRUNK_SCALAR_SHAPE_FUNCTION_GRAD_TPP_

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  ShapeFunctionScalarGrad<dim>::ShapeFunctionScalarGrad(
    const FiniteElement<dim> &                  fe,
    typename Triangulation<dim>::cell_iterator &cell,
    bool                                        verbose)
    : Function<dim>(dim)
    , fe_ptr(&fe)
    , dofs_per_cell(fe_ptr->dofs_per_cell)
    , shape_fun_index(0)
    , mapping(1)
    , current_cell_ptr(&cell)
    , grad(0)
    , verbose(verbose)
  {
    // Make sure the element is scalar.
    Assert(fe_ptr->n_components() == 1,
           ExcDimensionMismatch(1, fe_ptr->n_components()));
    if (verbose)
      {
        std::cout << "		Constructed scalar shape function gradient for "
                     "  "
                  << fe_ptr->get_name() << "   on cell   [";
        for (unsigned int i = 0; i < (std::pow(2, dim) - 1); ++i)
          {
            std::cout << cell->vertex(i) << ", \n";
          }
        std::cout << cell->vertex(std::pow(2, dim) - 1) << "]\n" << std::endl;
      }
  }

  template <int dim>
  void
    ShapeFunctionScalarGrad<dim>::set_current_cell(
      const typename Triangulation<dim>::cell_iterator &cell)
  {
    current_cell_ptr = &cell;
  }

  template <int dim>
  void
    ShapeFunctionScalarGrad<dim>::set_shape_fun_index(unsigned int index)
  {
    shape_fun_index = index;
  }

  template <int dim>
  void
    ShapeFunctionScalarGrad<dim>::vector_value(const Point<dim> &p,
                                               Vector<double> &  value) const
  {
    // Map physical points to reference cell
    Point<dim> point_on_ref_cell(
      mapping.transform_real_to_unit_cell(*current_cell_ptr, p));

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(point_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(*fe_ptr,
                            fake_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    (fe_values[grad].gradient(shape_fun_index, /* q_index */ 0)).unroll(value);
  }

  template <int dim>
  void
    ShapeFunctionScalarGrad<dim>::vector_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    const unsigned int n_q_points = points.size();

    // Map physical points to reference cell
    std::vector<Point<dim>> points_on_ref_cell(n_q_points);
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        points_on_ref_cell.at(i) =
          mapping.transform_real_to_unit_cell(*current_cell_ptr, points.at(i));
      }

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(points_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(*fe_ptr,
                            fake_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        (fe_values[grad].gradient(shape_fun_index,
                                  /* q_index */ i))
          .unroll(values.at(i));
      }
  }


  template <int dim>
  void
    ShapeFunctionScalarGrad<dim>::tensor_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<Tensor<1, dim>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    const unsigned int n_q_points = points.size();

    // Map physical points to reference cell
    std::vector<Point<dim>> points_on_ref_cell(n_q_points);
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        points_on_ref_cell.at(i) =
          mapping.transform_real_to_unit_cell(*current_cell_ptr, points.at(i));
      }

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(points_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(*fe_ptr,
                            fake_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        values.at(i) = fe_values[grad].gradient(shape_fun_index,
                                                /* q_index */ i);
      }
  }

} // namespace ShapeFun

#endif /* INCLUDE_TRUNK_SCALAR_SHAPE_FUNCTION_GRAD_TPP_ */
