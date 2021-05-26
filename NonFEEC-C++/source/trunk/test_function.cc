#include <trunk/test_function.h>

namespace ShapeFun
{
  TestFunction::TestFunction()
  {}

  void
    TestFunction::vector_value(const Point<3> &p, Vector<double> &value) const
  {
    value(0) = scale * p(0) * p(0) * p(1) * p(2) + 1;
    value(1) = scale * p(0) * p(1) * p(1) * p(2) + 1;
    value(2) = scale * p(0) * p(1) * p(2) * p(2) + 1;
  }

  void
    TestFunction::vector_value_list(const std::vector<Point<3>> &points,
                                    std::vector<Vector<double>> &values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));
    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p](0) =
          scale * points[p](0) * points[p](0) * points[p](1) * points[p](2) + 1;
        values[p](1) =
          scale * points[p](0) * points[p](1) * points[p](1) * points[p](2) + 1;
        values[p](2) =
          scale * points[p](0) * points[p](1) * points[p](2) * points[p](2) + 1;
      }
  }

  void
    TestFunction::tensor_value_list(const std::vector<Point<3>> &points,
                                    std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p][0] =
          scale * points[p](0) * points[p](0) * points[p](1) * points[p](2) + 1;
        values[p][1] =
          scale * points[p](0) * points[p](1) * points[p](1) * points[p](2) + 1;
        values[p][2] =
          scale * points[p](0) * points[p](1) * points[p](2) * points[p](2) + 1;
      }
  }


  /////////////////////////////////////////


  TestFunctionCurl::TestFunctionCurl()
  {}

  void
    TestFunctionCurl::vector_value(const Point<3> &p,
                                   Vector<double> &value) const
  {
    value(0) = scale * p(0) * (p(2) * p(2) - p(1) * p(1));
    value(1) = scale * p(1) * (p(0) * p(0) - p(2) * p(2));
    value(2) = scale * p(2) * (p(1) * p(1) - p(0) * p(0));
  }

  void
    TestFunctionCurl::vector_value_list(
      const std::vector<Point<3>> &points,
      std::vector<Vector<double>> &values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p](0) =
          scale * points[p](0) *
          (points[p](2) * points[p](2) - points[p](1) * points[p](1));
        values[p](1) =
          scale * points[p](1) *
          (points[p](0) * points[p](0) - points[p](2) * points[p](2));
        values[p](2) =
          scale * points[p](2) *
          (points[p](1) * points[p](1) - points[p](0) * points[p](0));
      }
  }

  void
    TestFunctionCurl::tensor_value_list(const std::vector<Point<3>> &points,
                                        std::vector<Tensor<1, 3>> &values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p][0] =
          scale * points[p](0) *
          (points[p](2) * points[p](2) - points[p](1) * points[p](1));
        values[p][1] =
          scale * points[p](1) *
          (points[p](0) * points[p](0) - points[p](2) * points[p](2));
        values[p][2] =
          scale * points[p](2) *
          (points[p](1) * points[p](1) - points[p](0) * points[p](0));
      }
  }

} // end namespace ShapeFun
