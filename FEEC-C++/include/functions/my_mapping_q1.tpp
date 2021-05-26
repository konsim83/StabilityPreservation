#ifndef INCLUDE_FUNCTIONS_MY_MAPPING_Q1_TPP_
#define INCLUDE_FUNCTIONS_MY_MAPPING_Q1_TPP_

#include <functions/my_mapping_q1.h>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  MyMappingQ1<dim>::MyMappingQ1(const MyMappingQ1<dim> &mapping)
    : coeff_matrix(mapping.coeff_matrix)
    , coeff_matrix_unit_cell(mapping.coeff_matrix_unit_cell)
    , cell_vertex(mapping.cell_vertex)
  {}


  /*
   * **************************
   * *** 2D implementations ***
   * **************************
   */

  template <>
  MyMappingQ1<2>::MyMappingQ1(
    const typename Triangulation<2>::active_cell_iterator &cell)
    : coeff_matrix(4, 4)
    , coeff_matrix_unit_cell(4, 4)
    , cell_vertex(4)
  {
    FullMatrix<double> point_matrix(4, 4);

    for (unsigned int alpha = 0; alpha < 4; ++alpha)
      {
        cell_vertex[alpha] = cell->vertex(alpha);

        // point matrix to be inverted
        point_matrix(0, alpha) = 1;
        point_matrix(1, alpha) = cell_vertex[alpha](0);
        point_matrix(2, alpha) = cell_vertex[alpha](1);
        point_matrix(3, alpha) = cell_vertex[alpha](0) * cell_vertex[alpha](1);
      }

    /*
     * Rows of coeff_matrix are the coefficients of the basis on the physical
     * cell
     */
    coeff_matrix.invert(point_matrix);


    /*
     * Coefficient matrix for unit cell
     */
    coeff_matrix_unit_cell(0, 0) = 1;
    coeff_matrix_unit_cell(0, 1) = -1;
    coeff_matrix_unit_cell(0, 2) = -1;
    coeff_matrix_unit_cell(0, 3) = 1;

    coeff_matrix_unit_cell(1, 0) = 0;
    coeff_matrix_unit_cell(1, 1) = 1;
    coeff_matrix_unit_cell(1, 2) = 0;
    coeff_matrix_unit_cell(1, 3) = -1;

    coeff_matrix_unit_cell(2, 0) = 0;
    coeff_matrix_unit_cell(2, 1) = 0;
    coeff_matrix_unit_cell(2, 2) = 1;
    coeff_matrix_unit_cell(2, 3) = -1;

    coeff_matrix_unit_cell(3, 0) = 0;
    coeff_matrix_unit_cell(3, 1) = 0;
    coeff_matrix_unit_cell(3, 2) = 0;
    coeff_matrix_unit_cell(3, 3) = 1;
  }


  template <>
  Point<2>
    MyMappingQ1<2>::map_real_to_unit_cell(const Point<2> &p) const
  {
    Point<2> p_out;

    for (unsigned int alpha = 0; alpha < 4; ++alpha)
      {
        const Point<2> &p_ref = GeometryInfo<2>::unit_cell_vertex(alpha);

        p_out += (coeff_matrix(alpha, 0) + coeff_matrix(alpha, 1) * p(0) +
                  coeff_matrix(alpha, 2) * p(1) +
                  coeff_matrix(alpha, 3) * p(0) * p(1)) *
                 p_ref;
      }

    return p_out;
  }


  template <>
  Point<2>
    MyMappingQ1<2>::map_unit_cell_to_real(const Point<2> &p) const
  {
    Point<2> p_out;

    return p_out = cell_vertex[0] * (1 - p(0)) * (1 - p(1)) +
                   cell_vertex[1] * p(0) * (1 - p(1)) +
                   cell_vertex[2] * (1 - p(0)) * p(1) +
                   cell_vertex[3] * p(0) * p(1);
  }


  template <>
  void
    MyMappingQ1<2>::map_real_to_unit_cell(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        points_out[i].clear();
        for (unsigned int alpha = 0; alpha < 4; ++alpha)
          {
            const Point<2> &p_ref = GeometryInfo<2>::unit_cell_vertex(alpha);

            points_out[i] +=
              (coeff_matrix(alpha, 0) +
               coeff_matrix(alpha, 1) * points_in[i](0) +
               coeff_matrix(alpha, 2) * points_in[i](1) +
               coeff_matrix(alpha, 3) * points_in[i](0) * points_in[i](1)) *
              p_ref;
          }
      }
  }


  template <>
  void
    MyMappingQ1<2>::map_unit_cell_to_real(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        points_out[i] =
          cell_vertex[0] * (1 - points_in[i](0)) * (1 - points_in[i](1)) +
          cell_vertex[1] * points_in[i](0) * (1 - points_in[i](1)) +
          cell_vertex[2] * (1 - points_in[i](0)) * points_in[i](1) +
          cell_vertex[3] * points_in[i](0) * points_in[i](1);
      }
  }


  template <>
  FullMatrix<double>
    MyMappingQ1<2>::jacobian_map_real_to_unit_cell(const Point<2> &p) const
  {
    FullMatrix<double> jacobian(2, 2);

    for (unsigned int alpha = 0; alpha < 4; ++alpha)
      {
        const Point<2> &p_ref = GeometryInfo<2>::unit_cell_vertex(alpha);

        const Point<2> grad_phi_alpha(coeff_matrix(alpha, 1) +
                                        coeff_matrix(alpha, 3) * p(1),
                                      coeff_matrix(alpha, 2) +
                                        coeff_matrix(alpha, 3) * p(0));

        for (unsigned int k = 0; k < 2; ++k)
          for (unsigned int l = 0; l < 2; ++l)
            jacobian(k, l) += p_ref(k) * grad_phi_alpha(l);
      }

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<2>::jacobian_map_real_to_unit_cell(
      const std::vector<Point<2>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == jacobian_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        jacobian_out[i] = 0;
        for (unsigned int alpha = 0; alpha < 4; ++alpha)
          {
            const Point<2> &p_ref = GeometryInfo<2>::unit_cell_vertex(alpha);

            const Point<2> grad_phi_alpha(
              coeff_matrix(alpha, 1) + coeff_matrix(alpha, 3) * points_in[i](1),
              coeff_matrix(alpha, 2) +
                coeff_matrix(alpha, 3) * points_in[i](0));

            for (unsigned int k = 0; k < 2; ++k)
              for (unsigned int l = 0; l < 2; ++l)
                jacobian_out[i](k, l) += p_ref(k) * grad_phi_alpha(l);
          }
      }
  }



  template <>
  FullMatrix<double>
    MyMappingQ1<2>::jacobian_map_unit_cell_to_real(const Point<2> &p) const
  {
    FullMatrix<double> jacobian(2, 2);



    for (unsigned int alpha = 0; alpha < 4; ++alpha)
      {
        const Point<2> grad_phi_alpha(coeff_matrix_unit_cell(alpha, 1) +
                                        coeff_matrix_unit_cell(alpha, 3) * p(1),
                                      coeff_matrix_unit_cell(alpha, 2) +
                                        coeff_matrix_unit_cell(alpha, 3) *
                                          p(0));

        for (unsigned int k = 0; k < 2; ++k)
          for (unsigned int l = 0; l < 2; ++l)
            jacobian(k, l) += cell_vertex[alpha](k) * grad_phi_alpha(l);
      }

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<2>::jacobian_map_unit_cell_to_real(
      const std::vector<Point<2>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == jacobian_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        jacobian_out[i] = 0;
        for (unsigned int alpha = 0; alpha < 4; ++alpha)
          {
            const Point<2> grad_phi_alpha(coeff_matrix_unit_cell(alpha, 1) +
                                            coeff_matrix_unit_cell(alpha, 3) *
                                              points_in[i](1),
                                          coeff_matrix_unit_cell(alpha, 2) +
                                            coeff_matrix_unit_cell(alpha, 3) *
                                              points_in[i](0));

            for (unsigned int k = 0; k < 2; ++k)
              for (unsigned int l = 0; l < 2; ++l)
                jacobian_out[i](k, l) +=
                  cell_vertex[alpha](k) * grad_phi_alpha(l);
          }
      }
  }



  /*
   * **************************
   * *** 3D implementations ***
   * **************************
   */


  template <>
  MyMappingQ1<3>::MyMappingQ1(
    const typename Triangulation<3>::active_cell_iterator &cell)
    : coeff_matrix(8, 8)
    , coeff_matrix_unit_cell(8, 8)
    , cell_vertex(8)
  {
    FullMatrix<double> point_matrix(8, 8);

    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        cell_vertex[alpha] = cell->vertex(alpha);

        // point matrix to be inverted
        point_matrix(0, alpha) = 1;
        point_matrix(1, alpha) = cell_vertex[alpha](0);
        point_matrix(2, alpha) = cell_vertex[alpha](1);
        point_matrix(3, alpha) = cell_vertex[alpha](2);
        point_matrix(4, alpha) = cell_vertex[alpha](0) * cell_vertex[alpha](1);
        point_matrix(5, alpha) = cell_vertex[alpha](1) * cell_vertex[alpha](2);
        point_matrix(6, alpha) = cell_vertex[alpha](0) * cell_vertex[alpha](2);
        point_matrix(7, alpha) =
          cell_vertex[alpha](0) * cell_vertex[alpha](1) * cell_vertex[alpha](2);
      }

    /*
     * Rows of coeff_matrix are the coefficients of the basis on the physical
     * cell
     */
    coeff_matrix.invert(point_matrix);

    ///////////////////

    point_matrix = 0;

    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        const Point<3> &p_ref = GeometryInfo<3>::unit_cell_vertex(alpha);

        // point matrix to be inverted
        point_matrix(0, alpha) = 1;
        point_matrix(1, alpha) = p_ref(0);
        point_matrix(2, alpha) = p_ref(1);
        point_matrix(3, alpha) = p_ref(2);
        point_matrix(4, alpha) = p_ref(0) * p_ref(1);
        point_matrix(5, alpha) = p_ref(1) * p_ref(2);
        point_matrix(6, alpha) = p_ref(0) * p_ref(2);
        point_matrix(7, alpha) = p_ref(0) * p_ref(1) * p_ref(2);
      }

    /*
     * Rows of coeff_matrix are the coefficients of the basis on the unit
     * cell
     */
    coeff_matrix_unit_cell.invert(point_matrix);
  }


  template <>
  Point<3>
    MyMappingQ1<3>::map_real_to_unit_cell(const Point<3> &p) const
  {
    Point<3> p_out;

    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        const Point<3> &p_ref = GeometryInfo<3>::unit_cell_vertex(alpha);

        p_out +=
          (coeff_matrix(alpha, 0) + coeff_matrix(alpha, 1) * p(0) +
           coeff_matrix(alpha, 2) * p(1) + coeff_matrix(alpha, 3) * p(2) +
           coeff_matrix(alpha, 4) * p(0) * p(1) +
           coeff_matrix(alpha, 5) * p(1) * p(2) +
           coeff_matrix(alpha, 6) * p(0) * p(2) +
           coeff_matrix(alpha, 7) * p(0) * p(1) * p(2)) *
          p_ref;
      }

    return p_out;
  }


  template <>
  Point<3>
    MyMappingQ1<3>::map_unit_cell_to_real(const Point<3> &p) const
  {
    Point<3> p_out;

    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        p_out += (coeff_matrix_unit_cell(alpha, 0) +
                  coeff_matrix_unit_cell(alpha, 1) * p(0) +
                  coeff_matrix_unit_cell(alpha, 2) * p(1) +
                  coeff_matrix_unit_cell(alpha, 3) * p(2) +
                  coeff_matrix_unit_cell(alpha, 4) * p(0) * p(1) +
                  coeff_matrix_unit_cell(alpha, 5) * p(1) * p(2) +
                  coeff_matrix_unit_cell(alpha, 6) * p(0) * p(2) +
                  coeff_matrix_unit_cell(alpha, 7) * p(0) * p(1) * p(2)) *
                 cell_vertex[alpha];
      }

    return p_out;
  }



  template <>
  void
    MyMappingQ1<3>::map_real_to_unit_cell(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        points_out[i].clear();
        for (unsigned int alpha = 0; alpha < 8; ++alpha)
          {
            const Point<3> &p_ref = GeometryInfo<3>::unit_cell_vertex(alpha);

            points_out[i] +=
              (coeff_matrix(alpha, 0) +
               coeff_matrix(alpha, 1) * points_in[i](0) +
               coeff_matrix(alpha, 2) * points_in[i](1) +
               coeff_matrix(alpha, 3) * points_in[i](2) +
               coeff_matrix(alpha, 4) * points_in[i](0) * points_in[i](1) +
               coeff_matrix(alpha, 5) * points_in[i](1) * points_in[i](2) +
               coeff_matrix(alpha, 6) * points_in[i](0) * points_in[i](2) +
               coeff_matrix(alpha, 7) * points_in[i](0) * points_in[i](1) *
                 points_in[i](2)) *
              p_ref;
          }
      }
  }


  template <>
  void
    MyMappingQ1<3>::map_unit_cell_to_real(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        points_out[i].clear();
        for (unsigned int alpha = 0; alpha < 8; ++alpha)
          {
            points_out[i] +=
              (coeff_matrix_unit_cell(alpha, 0) +
               coeff_matrix_unit_cell(alpha, 1) * points_in[i](0) +
               coeff_matrix_unit_cell(alpha, 2) * points_in[i](1) +
               coeff_matrix_unit_cell(alpha, 3) * points_in[i](2) +
               coeff_matrix_unit_cell(alpha, 4) * points_in[i](0) *
                 points_in[i](1) +
               coeff_matrix_unit_cell(alpha, 5) * points_in[i](1) *
                 points_in[i](2) +
               coeff_matrix_unit_cell(alpha, 6) * points_in[i](0) *
                 points_in[i](2) +
               coeff_matrix_unit_cell(alpha, 7) * points_in[i](0) *
                 points_in[i](1) * points_in[i](2)) *
              cell_vertex[alpha];
          }
      }
  }


  template <>
  FullMatrix<double>
    MyMappingQ1<3>::jacobian_map_real_to_unit_cell(const Point<3> &p) const
  {
    FullMatrix<double> jacobian(3, 3);

    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        const Point<3> &p_ref = GeometryInfo<3>::unit_cell_vertex(alpha);

        const Point<3> grad_phi_alpha(
          coeff_matrix(alpha, 1) + coeff_matrix(alpha, 4) * p(1) +
            coeff_matrix(alpha, 6) * p(2) +
            coeff_matrix(alpha, 7) * p(1) * p(2),
          coeff_matrix(alpha, 2) + coeff_matrix(alpha, 4) * p(0) +
            coeff_matrix(alpha, 5) * p(2) +
            coeff_matrix(alpha, 7) * p(0) * p(2),
          coeff_matrix(alpha, 3) + coeff_matrix(alpha, 5) * p(1) +
            coeff_matrix(alpha, 6) * p(0) +
            coeff_matrix(alpha, 7) * p(0) * p(1));

        for (unsigned int k = 0; k < 3; ++k)
          for (unsigned int l = 0; l < 3; ++l)
            jacobian(k, l) += p_ref(k) * grad_phi_alpha(l);
      }

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<3>::jacobian_map_real_to_unit_cell(
      const std::vector<Point<3>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == jacobian_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        jacobian_out[i] = 0;
        for (unsigned int alpha = 0; alpha < 8; ++alpha)
          {
            const Point<3> &p_ref = GeometryInfo<3>::unit_cell_vertex(alpha);

            const Point<3> grad_phi_alpha(
              coeff_matrix(alpha, 1) +
                coeff_matrix(alpha, 4) * points_in[i](1) +
                coeff_matrix(alpha, 6) * points_in[i](2) +
                coeff_matrix(alpha, 7) * points_in[i](1) * points_in[i](2),
              coeff_matrix(alpha, 2) +
                coeff_matrix(alpha, 4) * points_in[i](0) +
                coeff_matrix(alpha, 5) * points_in[i](2) +
                coeff_matrix(alpha, 7) * points_in[i](0) * points_in[i](2),
              coeff_matrix(alpha, 3) +
                coeff_matrix(alpha, 5) * points_in[i](1) +
                coeff_matrix(alpha, 6) * points_in[i](0) +
                coeff_matrix(alpha, 7) * points_in[i](0) * points_in[i](1));

            for (unsigned int k = 0; k < 3; ++k)
              for (unsigned int l = 0; l < 3; ++l)
                jacobian_out[i](k, l) += p_ref(k) * grad_phi_alpha(l);
          }
      }
  }

  template <>
  FullMatrix<double>
    MyMappingQ1<3>::jacobian_map_unit_cell_to_real(const Point<3> &p) const
  {
    FullMatrix<double> jacobian(3, 3);

    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        const Point<3> grad_phi_alpha(
          coeff_matrix_unit_cell(alpha, 1) +
            coeff_matrix_unit_cell(alpha, 4) * p(1) +
            coeff_matrix_unit_cell(alpha, 6) * p(2) +
            coeff_matrix_unit_cell(alpha, 7) * p(1) * p(2),
          coeff_matrix_unit_cell(alpha, 2) +
            coeff_matrix_unit_cell(alpha, 4) * p(0) +
            coeff_matrix_unit_cell(alpha, 5) * p(2) +
            coeff_matrix_unit_cell(alpha, 7) * p(0) * p(2),
          coeff_matrix_unit_cell(alpha, 3) +
            coeff_matrix_unit_cell(alpha, 5) * p(1) +
            coeff_matrix_unit_cell(alpha, 6) * p(0) +
            coeff_matrix_unit_cell(alpha, 7) * p(0) * p(1));

        for (unsigned int k = 0; k < 3; ++k)
          for (unsigned int l = 0; l < 3; ++l)
            jacobian(k, l) += cell_vertex[alpha](k) * grad_phi_alpha(l);
      }

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<3>::jacobian_map_unit_cell_to_real(
      const std::vector<Point<3>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == jacobian_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        jacobian_out[i] = 0;

        for (unsigned int alpha = 0; alpha < 8; ++alpha)
          {
            const Point<3> grad_phi_alpha(
              coeff_matrix_unit_cell(alpha, 1) +
                coeff_matrix_unit_cell(alpha, 4) * points_in[i](1) +
                coeff_matrix_unit_cell(alpha, 6) * points_in[i](2) +
                coeff_matrix_unit_cell(alpha, 7) * points_in[i](1) *
                  points_in[i](2),
              coeff_matrix_unit_cell(alpha, 2) +
                coeff_matrix_unit_cell(alpha, 4) * points_in[i](0) +
                coeff_matrix_unit_cell(alpha, 5) * points_in[i](2) +
                coeff_matrix_unit_cell(alpha, 7) * points_in[i](0) *
                  points_in[i](2),
              coeff_matrix_unit_cell(alpha, 3) +
                coeff_matrix_unit_cell(alpha, 5) * points_in[i](1) +
                coeff_matrix_unit_cell(alpha, 6) * points_in[i](0) +
                coeff_matrix_unit_cell(alpha, 7) * points_in[i](0) *
                  points_in[i](1));

            for (unsigned int k = 0; k < 3; ++k)
              for (unsigned int l = 0; l < 3; ++l)
                jacobian_out[i](k, l) +=
                  cell_vertex[alpha](k) * grad_phi_alpha(l);
          }
      }
  }

} // namespace ShapeFun



#endif /* INCLUDE_FUNCTIONS_MY_MAPPING_Q1_TPP_ */
