#ifndef INCLUDE_FUNCTIONS_CONCATINATE_FUNCTIONS_H_
#define INCLUDE_FUNCTIONS_CONCATINATE_FUNCTIONS_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace ShapeFun
{
  using namespace dealii;

  /*!
   * @class ShapeFunctionConcatinateVector
   *
   * @brief Concatination of Function<dim> objects
   *
   * This class represents a vector function that is made of two
   * concatinated <code> Function<dim> </code> objects.
   *
   */
  template <int dim>
  class ShapeFunctionConcatinateVector : public Function<dim>
  {
  public:
    /*!
     * Constructor takes two function objects and concatinates
     * them to a vecor function.
     *
     * @param function1
     * @param function2
     */
    ShapeFunctionConcatinateVector(const Function<dim> &function1,
                                   const Function<dim> &function2);

    /*!
     * Value of concatinated function of a given component.
     *
     * @param p
     * @param component
     * @return
     */
    virtual double
      value(const Point<dim> &p, const unsigned int component) const override;

    /*!
     * Vector value of concatinated function.
     *
     * @param p
     * @param value
     */
    virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;

    /*!
     * Vector value list of concatinated function.
     *
     * @param points
     * @param values
     */
    virtual void
      vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>> &  values) const override;

  private:
    /*!
     * Smart pointer to first componenet of input.
     */
    SmartPointer<const Function<dim>> function_ptr1;

    /*!
     * Smart pointer to second componenet of input.
     */
    SmartPointer<const Function<dim>> function_ptr2;
  };

} // namespace ShapeFun

#include <functions/concatinate_functions.tpp>

#endif /* INCLUDE_FUNCTIONS_CONCATINATE_FUNCTIONS_H_ */
