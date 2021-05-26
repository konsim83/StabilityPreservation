// This program tests the functionality of my_vector_tools (parallel
// and serial projection on FE spaces).

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

// C++ STL
#include <iostream>

// My library
#include <vector_tools/my_vector_tools.h>
#include <vector_tools/my_vector_tools.tpp>

using namespace dealii;

///////////////////////////////////
///////////////////////////////////
template <int dim>
class MyVectorFunction : public TensorFunction<1, dim>
{
public:
  MyVectorFunction()
    : TensorFunction<1, dim>(){};

  void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>> &  values) const override;
};

template <int dim>
void
  MyVectorFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                    std::vector<Tensor<1, dim>> &  values) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int i = 0; i < values.size(); ++i)
    {
      values[i].clear();
      for (unsigned int d = 0; d < dim; ++d)
        values[i][d] = d + 1.0;
    }
}
///////////////////////////////////
///////////////////////////////////

///////////////////////////////////
///////////////////////////////////
int
  main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, dealii::numbers::invalid_unsigned_int);

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  const int dim = 3, degree = 0, n_refine = 3;

  ConditionalOStream pcout(std::cout,
                           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                            0));

  parallel::distributed::Triangulation<dim> triangulation(
    mpi_communicator,
    typename Triangulation<dim>::MeshSmoothing(
      Triangulation<dim>::smoothing_on_refinement |
      Triangulation<dim>::smoothing_on_coarsening));

  GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);
  triangulation.refine_global(n_refine);

  FE_Nedelec<dim> fe(degree);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // Quadrature used for projection
  QGauss<dim> quad_rule(/* order = */ 3);

  // Setup function
  MyVectorFunction<dim> my_vector_function;

  TrilinosWrappers::MPI::Vector projected_function;

  IndexSet locally_owned_dofs;
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  projected_function.reinit(locally_owned_dofs, mpi_communicator);

  try
    {
      MyVectorTools::project_on_fe_space(dof_handler,
                                         constraints,
                                         quad_rule,
                                         my_vector_function,
                                         projected_function,
                                         mpi_communicator);

      // Write only for process 0
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          IndexSet::ElementIterator local_index = locally_owned_dofs.begin(),
                                    local_index_end = locally_owned_dofs.end();

          std::cout << "Values of the projected vector in MPI process 0:"
                    << std::endl;

          for (; local_index != local_index_end; ++local_index)
            {
              std::cout << "   " << projected_function[*local_index]
                        << std::endl;
            }

          std::cout << "Projection test succeeded." << std::endl;
        }

      constraints.clear();
      dof_handler.clear();
    }
  catch (...)
    {
      //      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      std::cout << "Projection test failed." << std::endl;
    }
}
///////////////////////////////////
///////////////////////////////////
