#ifndef INCLUDE_FUNCTIONS_TEST_FUNCTION_H_
#define INCLUDE_FUNCTIONS_TEST_FUNCTION_H_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

// STL
#include <cmath>
#include <fstream>


namespace ShapeFun
{
  using namespace dealii;


  class TestFunction : public Function<3>
  {
  public:
    TestFunction();

    virtual void
      vector_value(const Point<3> &p, Vector<double> &value) const override;

    virtual void
      vector_value_list(const std::vector<Point<3>> &points,
                        std::vector<Vector<double>> &values) const override;

    void
      tensor_value_list(const std::vector<Point<3>> &points,
                        std::vector<Tensor<1, 3>> &  values) const;

    const double scale = 1000;
  };


  class TestFunctionCurl : public Function<3>
  {
  public:
    TestFunctionCurl();

    virtual void
      vector_value(const Point<3> &p, Vector<double> &value) const override;

    virtual void
      vector_value_list(const std::vector<Point<3>> &points,
                        std::vector<Vector<double>> &values) const override;

    void
      tensor_value_list(const std::vector<Point<3>> &points,
                        std::vector<Tensor<1, 3>> &  values) const;

    const double scale = 1000;
  };

} // namespace ShapeFun

#endif /* INCLUDE_FUNCTIONS_TEST_FUNCTION_H_ */
