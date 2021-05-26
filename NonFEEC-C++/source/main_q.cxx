// deal.ii parameter files
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>

// Std C++
#include <fstream>
#include <iostream>

// my headers
#include "Q/q_global.h"
#include "Q/q_ref.h"
#include "Utilities.h"

/**
 * Main file. Call ./main and see instructions for command lie parameters.
 */
int
  main(int argc, char *argv[])
{
  // Very simple way of input handling.
  if (argc < 2)
    {
      std::cout << "You must provide an input file \"-p <filename>\""
                << std::endl;
      exit(1);
    }

  std::string input_file = "";

  std::list<std::string> args;
  for (int i = 1; i < argc; ++i)
    {
      args.push_back(argv[i]);
    }

  while (args.size())
    {
      if (args.front() == std::string("-p"))
        {
          if (args.size() == 1) /* This is not robust. */
            {
              std::cerr << "Error: flag '-p' must be followed by the "
                        << "name of a parameter file." << std::endl;
              exit(1);
            }
          else
            {
              args.pop_front();
              input_file = args.front();
              args.pop_front();
            }
        }
      else
        {
          std::cerr << "Unknown command line option: " << args.front()
                    << std::endl;
          exit(1);
        }
    } // end while

  try
    {
#ifdef USE_PETSC_LA
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, /* disable threading for petsc */ 1);
#else
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);
#endif

      {
        dealii::deallog.depth_console(2);

        // reference solution
        Q::ParametersStd parameters(input_file);
        Q::QStd          standard_laplace_std(parameters, input_file);
        standard_laplace_std.run();
      }

      {
        dealii::deallog.depth_console(0);

        // multiscale solution
        Q::ParametersMs parameters(input_file);
        Q::QMultiscale  standard_laplace_global(parameters, input_file);
        standard_laplace_global.run();
      }
    } /* try */

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    } /* catch deal.ii exceptions */

  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    } /* catch all other exceptions */

  return 0;
}
