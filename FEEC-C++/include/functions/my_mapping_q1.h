#ifndef INCLUDE_FUNCTIONS_MY_MAPPING_Q1_H_
#define INCLUDE_FUNCTIONS_MY_MAPPING_Q1_H_

// Deal.ii
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// STL
#include <vector>

// My Headers

namespace ShapeFun
{
  using namespace dealii;

  /*!
   * @class MyMappingQ1
   *
   * @brief Implements \f$Q_1\f$-mappings.
   *
   * This class implements \f$Q_1\f$-mappings in a slightly different way than
   * deal.ii since this is more suitable for our use-case and speeds up things
   * significantly. It stores data for both the mapping from the unit cell to
   * the physical cell and back.
   */
  template <int dim>
  class MyMappingQ1
  {
  public:
    MyMappingQ1() = delete;

    /*!
     * Constructor.
     *
     * @param cell
     */
    MyMappingQ1(const typename Triangulation<dim>::active_cell_iterator &cell);

    /*!
     * Copy constructor.
     */
    MyMappingQ1(const MyMappingQ1<dim> &);

    /*!
     * Map a single point from the physical cell to the unit cell.
     *
     * @param p
     * @return
     */
    Point<dim>
      map_real_to_unit_cell(const Point<dim> &p) const;

    /*!
     * Map a list of points from the physical cell to the unit cell.
     *
     * @param points_in
     * @param points_out
     */
    void
      map_real_to_unit_cell(const std::vector<Point<dim>> &points_in,
                            std::vector<Point<dim>> &      points_out) const;

    /*!
     * Map a single point from the unit cell to the physical cell.
     *
     * @param p
     * @return
     */
    Point<dim>
      map_unit_cell_to_real(const Point<dim> &p) const;

    /*!
     * Map a list of points from the unit cell to the physical cell.
     *
     * @param points_in
     * @param points_out
     */
    void
      map_unit_cell_to_real(const std::vector<Point<dim>> &points_in,
                            std::vector<Point<dim>> &      points_out) const;

    /*!
     * Get the Jacobian of the map from the physical cell to the unit cell for a
     * single point.
     *
     * @param p
     * @return
     */
    FullMatrix<double>
      jacobian_map_real_to_unit_cell(const Point<dim> &p) const;

    /*!
     * Get the Jacobian of the map from the physical cell to the unit cell for a
     * list of points.
     *
     * @param points_in
     * @param jacobian_out
     */
    void
      jacobian_map_real_to_unit_cell(
        const std::vector<Point<dim>> &  points_in,
        std::vector<FullMatrix<double>> &jacobian_out) const;

    /*!
     * Get the Jacobian of the map from the unit cell to the physical cell for a
     * single point.
     *
     * @param p
     * @return
     */
    FullMatrix<double>
      jacobian_map_unit_cell_to_real(const Point<dim> &p) const;

    /*!
     * Get the Jacobian of the map from the unit cell to the physical cell for a
     * list of points.
     *
     * @param points_in
     * @param jacobian_out
     */
    void
      jacobian_map_unit_cell_to_real(
        const std::vector<Point<dim>> &  points_in,
        std::vector<FullMatrix<double>> &jacobian_out) const;

  private:
    /*!
     * Matrix holds coefficients of basis on physical cell.
     */
    FullMatrix<double> coeff_matrix;

    /*!
     * Matrix holds coefficients of basis on unit cell.
     */
    FullMatrix<double> coeff_matrix_unit_cell;

    std::vector<Point<dim>> cell_vertex;
  };


  /*
   * 2D declarations of specializations
   */
  template <>
  MyMappingQ1<2>::MyMappingQ1(
    const typename Triangulation<2>::active_cell_iterator &cell);

  template <>
  Point<2>
    MyMappingQ1<2>::map_real_to_unit_cell(const Point<2> &p) const;

  template <>
  void
    MyMappingQ1<2>::map_real_to_unit_cell(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &      points_out) const;

  template <>
  Point<2>
    MyMappingQ1<2>::map_unit_cell_to_real(const Point<2> &p) const;

  template <>
  void
    MyMappingQ1<2>::map_unit_cell_to_real(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &      points_out) const;

  template <>
  FullMatrix<double>
    MyMappingQ1<2>::jacobian_map_real_to_unit_cell(const Point<2> &p) const;

  template <>
  void
    MyMappingQ1<2>::jacobian_map_real_to_unit_cell(
      const std::vector<Point<2>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const;

  template <>
  FullMatrix<double>
    MyMappingQ1<2>::jacobian_map_unit_cell_to_real(const Point<2> &p) const;

  template <>
  void
    MyMappingQ1<2>::jacobian_map_unit_cell_to_real(
      const std::vector<Point<2>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const;

  /*
   * 3D declarations of specializations
   */
  template <>
  MyMappingQ1<3>::MyMappingQ1(
    const typename Triangulation<3>::active_cell_iterator &cell);

  template <>
  Point<3>
    MyMappingQ1<3>::map_real_to_unit_cell(const Point<3> &p) const;

  template <>
  void
    MyMappingQ1<3>::map_real_to_unit_cell(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &      points_out) const;

  template <>
  Point<3>
    MyMappingQ1<3>::map_unit_cell_to_real(const Point<3> &p) const;

  template <>
  void
    MyMappingQ1<3>::map_unit_cell_to_real(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &      points_out) const;

  template <>
  FullMatrix<double>
    MyMappingQ1<3>::jacobian_map_real_to_unit_cell(const Point<3> &p) const;

  template <>
  void
    MyMappingQ1<3>::jacobian_map_real_to_unit_cell(
      const std::vector<Point<3>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const;

  template <>
  FullMatrix<double>
    MyMappingQ1<3>::jacobian_map_unit_cell_to_real(const Point<3> &p) const;

  template <>
  void
    MyMappingQ1<3>::jacobian_map_unit_cell_to_real(
      const std::vector<Point<3>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const;


  /*
   * exernal template instantiations
   */
  extern template class MyMappingQ1<2>;
  extern template class MyMappingQ1<3>;

} // namespace ShapeFun



#endif /* INCLUDE_FUNCTIONS_MY_MAPPING_Q1_H_ */
