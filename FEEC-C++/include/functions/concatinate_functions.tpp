#ifndef INCLUDE_FUNCTIONS_CONCATINATE_FUNCTIONS_TPP_
#define INCLUDE_FUNCTIONS_CONCATINATE_FUNCTIONS_TPP_

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  ShapeFunctionConcatinateVector<dim>::ShapeFunctionConcatinateVector(
    const Function<dim> &function1,
    const Function<dim> &function2)
    : Function<dim>(function1.n_components + function2.n_components)
    , function_ptr1(&function1)
    , function_ptr2(&function2)
  {}

  template <int dim>
  double
    ShapeFunctionConcatinateVector<dim>::value(
      const Point<dim> & p,
      const unsigned int component) const
  {
    if (component < function_ptr1->n_components)
      {
        Vector<double> value1(function_ptr1->n_components);
        function_ptr1->vector_value(p, value1);
        return value1(component);
      }
    else
      {
        Vector<double> value2(function_ptr2->n_components);
        function_ptr2->vector_value(p, value2);
        return value2(component - function_ptr1->n_components);
      }
  }

  template <int dim>
  void
    ShapeFunctionConcatinateVector<dim>::vector_value(
      const Point<dim> &p,
      Vector<double> &  value) const
  {
    Vector<double> value1(function_ptr1->n_components);
    function_ptr1->vector_value(p, value1);

    Vector<double> value2(function_ptr2->n_components);
    function_ptr2->vector_value(p, value2);

    for (unsigned int j = 0; j < function_ptr1->n_components; ++j)
      value(j) = value1(j);
    for (unsigned int j = 0; j < function_ptr2->n_components; ++j)
      value(function_ptr1->n_components + j) = value2(j);
  }

  template <int dim>
  void
    ShapeFunctionConcatinateVector<dim>::vector_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    Vector<double> value1(function_ptr1->n_components);
    Vector<double> value2(function_ptr2->n_components);

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        Assert(values[i].size() ==
                 (function_ptr1->n_components + function_ptr2->n_components),
               ExcDimensionMismatch(values[i].size(),
                                    (function_ptr1->n_components +
                                     function_ptr2->n_components)));

        value1 = 0;
        value2 = 0;
        function_ptr1->vector_value(points[i], value1);
        function_ptr2->vector_value(points[i], value2);

        for (unsigned int j = 0; j < function_ptr1->n_components; ++j)
          {
            values[i](j) = value1(j);
          }
        for (unsigned int j = 0; j < function_ptr2->n_components; ++j)
          {
            values[i](function_ptr1->n_components + j) = value2(j);
          }
      }
  }

} // namespace ShapeFun

#endif /* INCLUDE_FUNCTIONS_CONCATINATE_FUNCTIONS_TPP_ */
