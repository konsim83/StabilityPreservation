#ifndef INCLUDE_TRUNK_SCALAR_SHAPE_FUNCTION_GRAD_H_
#define INCLUDE_TRUNK_SCALAR_SHAPE_FUNCTION_GRAD_H_

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
   * @class ShapeFunctionScalarGrad
   *
   * @brief Class for evaluations of vector valued shape functions.
   *
   * @note This is slow. Only use for quick and dirty prototyping and sanity checks.
   */
  template <int dim>
  class ShapeFunctionScalarGrad : public Function<dim>
  {
  public:
    /*!
     * Constructor takes a vector finite element like <code>BDM<dim> <\code>
     * or <code> RaviartThomas<dim> <\code> and a cell iterator pointing to
     * a certain cell in a triangulation.
     *
     * @param fe
     * @param cell
     * @param verbose = false
     */
    ShapeFunctionScalarGrad(const FiniteElement<dim> &                  fe,
                            typename Triangulation<dim>::cell_iterator &cell,
                            bool verbose = false);

    /*!
     * Evaluate shape function at point <code> p<\code>
     *
     * @param[in] p
     * @param[out] value
     */
    virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;

    /*!
     * Evaluate shape function at point list <code> points <\code>
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>> &  values) const override;

    /*!
     * Evaluate shape function at point list <code> points <\code>
     *
     * @param[in] points
     * @param[out] values
     */
    void
      tensor_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Tensor<1, dim>> &  values) const;

    /*!
     * Set pointer to current cell (actually and iterator).
     *
     * @param cell
     */
    void
      set_current_cell(const typename Triangulation<dim>::cell_iterator &cell);

    /*!
     * Set shape function index.
     *
     * @param index
     */
    void
      set_shape_fun_index(unsigned int index);

  private:
    SmartPointer<const FiniteElement<dim>> fe_ptr;
    const unsigned int                     dofs_per_cell;
    unsigned int                           shape_fun_index;

    const MappingQ<dim> mapping;

    typename Triangulation<dim>::cell_iterator *current_cell_ptr;

    const FEValuesExtractors::Scalar grad;

    const bool verbose;
  };

} // namespace ShapeFun

#include <trunk/scalar_shape_function_grad.tpp>

#endif /* INCLUDE_TRUNK_SCALAR_SHAPE_FUNCTION_GRAD_H_ */
