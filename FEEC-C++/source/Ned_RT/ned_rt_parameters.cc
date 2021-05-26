#include <Ned_RT/ned_rt_parameters.h>

namespace NedRT
{
  using namespace dealii;

  ParametersStd::ParametersStd(const std::string &parameter_filename)
    : compute_solution(true)
    , verbose(true)
    , use_direct_solver(false)
    , renumber_dofs(true)
    , n_refine(3)
    , transfer_to_level(3)
    , filename_output("NED_RT_Std")
    , dirname_output("NED_RT")
    , use_exact_solution(false)
  {
    ParameterHandler prm;

    ParametersStd::declare_parameters(prm);

    std::ifstream parameter_file(parameter_filename);
    if (!parameter_file)
      {
        parameter_file.close();
        std::ofstream parameter_out(parameter_filename);
        prm.print_parameters(parameter_out, ParameterHandler::Text);
        AssertThrow(
          false,
          ExcMessage(
            "Input parameter file <" + parameter_filename +
            "> not found. Creating a template file of the same name."));
      }

    prm.parse_input(parameter_file,
                    /* filename = */ "generated_parameter.in",
                    /* last_line = */ "",
                    /* skip_undefined = */ true);
    ParametersStd::parse_parameters(prm);
  }

  void
    ParametersStd::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Standard method parameters");
    {
      prm.enter_subsection("Mesh");
      {
        prm.declare_entry("refinements",
                          "3",
                          Patterns::Integer(1, 10),
                          "Number of initial mesh refinements.");
        prm.declare_entry("transfer to refinement level",
                          "3",
                          Patterns::Integer(1, 10),
                          "Transfer solution to a different refinement level.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Control flow");
      {
        prm.declare_entry("compute solution",
                          "true",
                          Patterns::Bool(),
                          "Choose whether to compute the solution or not.");
        prm.declare_entry("verbose",
                          "true",
                          Patterns::Bool(),
                          "Set runtime output true or false.");
        prm.declare_entry("use direct solver",
                          "true",
                          Patterns::Bool(),
                          "Use direct solvers true or false.");
        prm.declare_entry(
          "dof renumbering",
          "true",
          Patterns::Bool(),
          "Dof renumbering reduces bandwidth in system matrices.");
      }
      prm.leave_subsection();

      prm.declare_entry("filename output",
                        "NED_RT_Std",
                        Patterns::FileName(),
                        ".");
      prm.declare_entry("dirname output", "NED_RT", Patterns::FileName(), ".");

      prm.declare_entry("use exact solution",
                        "false",
                        Patterns::Bool(),
                        "Allows comparison with exact solution.");
    }
    prm.leave_subsection();
  }

  void
    ParametersStd::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Standard method parameters");
    {
      prm.enter_subsection("Mesh");
      {
        n_refine          = prm.get_integer("refinements");
        transfer_to_level = prm.get_integer("transfer to refinement level");
      }
      prm.leave_subsection();

      prm.enter_subsection("Control flow");
      {
        compute_solution  = prm.get_bool("compute solution");
        verbose           = prm.get_bool("verbose");
        use_direct_solver = prm.get_bool("use direct solver");
        renumber_dofs     = prm.get_bool("dof renumbering");
      }
      prm.leave_subsection();

      filename_output = prm.get("filename output");
      dirname_output  = prm.get("dirname output");

      use_exact_solution = prm.get_bool("use exact solution");
    }
    prm.leave_subsection();
  }

  ParametersMs::ParametersMs(const std::string &parameter_filename)
    : compute_solution(true)
    , verbose(true)
    , use_direct_solver(false)
    , renumber_dofs(true)
    , n_refine_global(2)
    , n_refine_local(2)
    , filename_output("NED_RT_Ms")
    , dirname_output("NED_RT")
    , use_exact_solution(false)
  {
    ParameterHandler prm;

    ParametersMs::declare_parameters(prm);

    std::ifstream parameter_file(parameter_filename);
    if (!parameter_file)
      {
        parameter_file.close();
        std::ofstream parameter_out(parameter_filename);
        prm.print_parameters(parameter_out, ParameterHandler::Text);
        AssertThrow(
          false,
          ExcMessage(
            "Input parameter file <" + parameter_filename +
            "> not found. Creating a template file of the same name."));
      }

    prm.parse_input(parameter_file,
                    /* filename = */ "generated_parameter.in",
                    /* last_line = */ "",
                    /* skip_undefined = */ true);
    ParametersMs::parse_parameters(prm);
  }

  void
    ParametersMs::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Multiscale method parameters");
    {
      prm.enter_subsection("Mesh");
      {
        prm.declare_entry("global refinements",
                          "2",
                          Patterns::Integer(1, 10),
                          "Number of initial coarse mesh refinements.");
        prm.declare_entry("local refinements",
                          "2",
                          Patterns::Integer(1, 10),
                          "Number of initial coarse mesh refinements.");
      }
      prm.leave_subsection();

      prm.enter_subsection("Control flow");
      {
        prm.declare_entry("compute solution",
                          "true",
                          Patterns::Bool(),
                          "Choose whether to compute the solution or not.");
        prm.declare_entry("verbose",
                          "true",
                          Patterns::Bool(),
                          "Set runtime output true or false.");
        prm.declare_entry("verbose basis",
                          "false",
                          Patterns::Bool(),
                          "Set runtime output true or false for basis.");
        prm.declare_entry("use direct solver",
                          "false",
                          Patterns::Bool(),
                          "Use direct solvers true or false.");
        prm.declare_entry("use direct solver basis",
                          "false",
                          Patterns::Bool(),
                          "Use direct solvers true or false.");
        prm.declare_entry(
          "dof renumbering",
          "true",
          Patterns::Bool(),
          "Dof renumbering reduces bandwidth in system matrices.");
        prm.declare_entry("write first basis",
                          "false",
                          Patterns::Bool(),
                          "Decide whether first cell's basis will be "
                          "written for diagnostic purposes.");
      }
      prm.leave_subsection();

      prm.declare_entry("filename output",
                        "NED_RT_Ms",
                        Patterns::FileName(),
                        ".");
      prm.declare_entry("dirname output", "NED_RT", Patterns::FileName(), ".");

      prm.declare_entry("use exact solution",
                        "false",
                        Patterns::Bool(),
                        "Allows comparison with exact solution.");
    }
    prm.leave_subsection();
  }

  void
    ParametersMs::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Multiscale method parameters");
    {
      prm.enter_subsection("Mesh");
      {
        n_refine_global = prm.get_integer("global refinements");
        n_refine_local  = prm.get_integer("local refinements");
      }
      prm.leave_subsection();

      prm.enter_subsection("Control flow");
      {
        compute_solution        = prm.get_bool("compute solution");
        verbose                 = prm.get_bool("verbose");
        verbose_basis           = prm.get_bool("verbose basis");
        use_direct_solver       = prm.get_bool("use direct solver");
        use_direct_solver_basis = prm.get_bool("use direct solver basis");
        renumber_dofs           = prm.get_bool("dof renumbering");
        prevent_output          = !prm.get_bool("write first basis");
      }
      prm.leave_subsection();

      filename_output = prm.get("filename output");
      dirname_output  = prm.get("dirname output");

      use_exact_solution = prm.get_bool("use exact solution");
    }
    prm.leave_subsection();
  }

  ParametersBasis::ParametersBasis(const ParametersMs &parameters_ms)
    : verbose(parameters_ms.verbose_basis)
    , use_direct_solver(parameters_ms.use_direct_solver_basis)
    , renumber_dofs(parameters_ms.renumber_dofs)
    , prevent_output(parameters_ms.prevent_output)
    , output_flag(false)
    , n_refine_global(parameters_ms.n_refine_global)
    , n_refine_local(parameters_ms.n_refine_local)
    , filename_global(parameters_ms.filename_output)
    , dirname_output(parameters_ms.dirname_output)
    , use_exact_solution(parameters_ms.use_exact_solution)
  {}

  ParametersBasis::ParametersBasis(const ParametersBasis &other)
    : verbose(other.verbose)
    , use_direct_solver(other.use_direct_solver)
    , renumber_dofs(other.renumber_dofs)
    , prevent_output(other.prevent_output)
    , output_flag(other.output_flag)
    , n_refine_global(other.n_refine_global)
    , n_refine_local(other.n_refine_local)
    , filename_global(other.filename_global)
    , dirname_output(other.dirname_output)
    , use_exact_solution(other.use_exact_solution)
  {}

  void
    ParametersBasis::set_output_flag(CellId local_cell_id, CellId first_cell)
  {
    if (!prevent_output)
      output_flag = (local_cell_id == first_cell);
  }

} // namespace NedRT
