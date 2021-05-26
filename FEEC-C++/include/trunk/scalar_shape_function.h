#ifndef SHAPE_FUN_SCALAR_TPP_
#define SHAPE_FUN_SCALAR_TPP_

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
   * @class ShapeFunctionScalar
   *
   * @brief Class for evaluations of scalar valued shape functions.
   *
   * @note This is slow. Only use for quick and dirty prototyping and sanity checks.
   */
  template <int dim>
  class ShapeFunctionScalar : public Function<dim>
  {
  public:
    /*!
     * Constructor takes a scalar finite element like <code>FE_Q<dim>
     * <\code> and a cell iterator pointing to a certain cell in a
     * triangulation.
     *
     * @param fe
     * @param cell
     * @param verbose = false
     */
    ShapeFunctionScalar(const FiniteElement<dim> &                         fe,
                        typename Triangulation<dim>::active_cell_iterator &cell,
                        bool verbose = false);

    /*!
     * Evaluate shape function at point <code> p<\code>
     *
     * @param[in] p
     * @param[in] component
     */
    virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

    /*!
     * Evaluate shape function at point list <code> points <\code>
     *
     * @param[in] points
     * @param[out] values
     */
    virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double> &          values,
                 const unsigned int             component = 0) const override;

    /*!
     * Set pointer to current cell (actually and iterator).
     *
     * @param cell
     */
    void
      set_current_cell(
        const typename Triangulation<dim>::active_cell_iterator &cell);

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

    typename Triangulation<dim>::active_cell_iterator *current_cell_ptr;

    const bool verbose;
  };

} // namespace ShapeFun

#include <trunk/scalar_shape_function.tpp>

#endif /* SHAPE_FUN_SCALAR_TPP_ */
