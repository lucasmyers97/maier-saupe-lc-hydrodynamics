#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <deal.II/lac/generic_linear_algebra.h>
#include <string>
#include <limits>
#include <exception>
#include <fstream>
#include <iostream>
#include <utility>
#include <memory>

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Utilities/Serialization.hpp"
#include "Utilities/DefectGridGenerator.hpp"
// #include "Utilities/git_version.hpp"

template <int dim>
NematicSystemMPIDriver<dim>::
NematicSystemMPIDriver(unsigned int degree_,
                       unsigned int num_refines_,
                       bool refine_further_flag_,
                       double left_,
                       double right_,
                       std::string grid_type_,
                       double dt_,
                       unsigned int n_steps_,
                       std::string time_discretization_,
                       double simulation_tol_,
                       double simulation_newton_step_,
                       unsigned int simulation_max_iters_,
                       double defect_size_,
                       double defect_charge_threshold_,
                       unsigned int vtu_interval_,
                       unsigned int checkpoint_interval_,
                       std::string data_folder_,
                       std::string config_filename_,
                       std::string defect_filename_,
                       std::string energy_filename_,
                       std::string archive_filename_)
    : mpi_communicator(MPI_COMM_WORLD)
    , tria(mpi_communicator,
           typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening))

    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      dealii::TimerOutput::summary,
                      dealii::TimerOutput::cpu_and_wall_times)

    , degree(degree_)
    , num_refines(num_refines_)
    , refine_further_flag(refine_further_flag_)
    , left(left_)
    , right(right_)
    , grid_type(grid_type_)

    , dt(dt_)
    , n_steps(n_steps_)

    , time_discretization(time_discretization_)
    , simulation_tol(simulation_tol_)
    , simulation_newton_step(simulation_newton_step_)
    , simulation_max_iters(simulation_max_iters_)

    , defect_size(defect_size_)
    , defect_charge_threshold(defect_charge_threshold_)

    , vtu_interval(vtu_interval_)
    , checkpoint_interval(checkpoint_interval_)
    , data_folder(data_folder_)
    , config_filename(config_filename_)
    , defect_filename(defect_filename_)
    , energy_filename(energy_filename_)
    , archive_filename(archive_filename_)
{}



template <int dim>
void NematicSystemMPIDriver<dim>::
declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("NematicSystemMPIDriver");

    prm.declare_entry("Finite element degree",
                      "1",
                      dealii::Patterns::Integer());
    prm.declare_entry("Number of refines",
                      "6",
                      dealii::Patterns::Integer());
    prm.declare_entry("Refine further flag",
                      "false",
                      dealii::Patterns::Bool());
    prm.declare_entry("Left",
                      "-1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Right",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Grid type",
                      "hypercube",
                      dealii::Patterns::Selection("hypercube|hyperball|two-defect-complement"));
    prm.declare_entry("Defect position",
                      "20.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Defect radius",
                      "2.5",
                      dealii::Patterns::Double());
    prm.declare_entry("Outer radius",
                      "5.0",
                      dealii::Patterns::Double());

    prm.declare_entry("dt",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Number of steps",
                      "30",
                      dealii::Patterns::Integer());

    prm.declare_entry("Time discretization",
                      "convex_splitting",
                      dealii::Patterns::Selection("convex_splitting"
                                                  "|forward_euler"));
    prm.declare_entry("Simulation tolerance",
                      "1e-10",
                      dealii::Patterns::Double());
    prm.declare_entry("Simulation newton step",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Simulation maximum iterations",
                      "20",
                      dealii::Patterns::Integer());

    prm.declare_entry("Defect size",
                      "2.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Defect charge threshold",
                      "0.3",
                      dealii::Patterns::Double());

    prm.declare_entry("Vtu interval",
                      "10",
                      dealii::Patterns::Integer());
    prm.declare_entry("Checkpoint interval",
                      "10",
                      dealii::Patterns::Integer());
    prm.declare_entry("Data folder",
                      "./",
                      dealii::Patterns::DirectoryName());
    prm.declare_entry("Configuration filename",
                      "nematic_configuration",
                      dealii::Patterns::FileName());
    prm.declare_entry("Defect filename",
                      "defect_positions",
                      dealii::Patterns::FileName());
    prm.declare_entry("Energy filename",
                      "configuration_energy",
                      dealii::Patterns::FileName());
    prm.declare_entry("Archive filename",
                      "nematic_simulation.ar",
                      dealii::Patterns::FileName());



    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPIDriver<dim>::
get_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("NematicSystemMPIDriver");

    degree = prm.get_integer("Finite element degree");
    num_refines = prm.get_integer("Number of refines");
    refine_further_flag = prm.get_bool("Refine further flag");
    left = prm.get_double("Left");
    right = prm.get_double("Right");
    grid_type = prm.get("Grid type");
    defect_position = prm.get_double("Defect position");
    defect_radius = prm.get_double("Defect radius");
    outer_radius = prm.get_double("Outer radius");

    dt = prm.get_double("dt");
    n_steps = prm.get_integer("Number of steps");

    time_discretization = prm.get("Time discretization");
    simulation_tol = prm.get_double("Simulation tolerance");
    simulation_newton_step = prm.get_double("Simulation newton step");
    simulation_max_iters = prm.get_integer("Simulation maximum iterations");

    defect_size = prm.get_double("Defect size");
    defect_charge_threshold = prm.get_double("Defect charge threshold");

    vtu_interval = prm.get_integer("Vtu interval");
    checkpoint_interval = prm.get_integer("Checkpoint interval");
    data_folder = prm.get("Data folder");
    config_filename = prm.get("Configuration filename");
    defect_filename = prm.get("Defect filename");
    energy_filename = prm.get("Energy filename");
    archive_filename = prm.get("Archive filename");

    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPIDriver<dim>::
print_parameters(std::string filename, dealii::ParameterHandler &prm)
{
    if (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
    {
        NematicSystemMPIDriver<dim>::declare_parameters(prm);
        NematicSystemMPI<dim>::declare_parameters(prm);
        prm.print_parameters(filename,
                             dealii::ParameterHandler::OutputStyle::PRM);
    }
}



template <int dim>
void NematicSystemMPIDriver<dim>::make_grid()
{
    if (grid_type == "hypercube")
    {
        dealii::GridGenerator::hyper_cube(tria, left, right);
    }
    else if (grid_type == "hyperball")
    {
        double midpoint = 0.5 * (right + left);
        double length = right - left;
        dealii::Point<dim> center;
        for (int i = 0; i < dim; ++i)
            center[i] = midpoint;
        double r = 0.5 * length;
        dealii::GridGenerator::hyper_ball_balanced(tria, center, r);
    }
    else if (grid_type == "two-defect-complement")
    {
        DefectGridGenerator::defect_mesh_complement(tria, 
                                                    defect_position,
                                                    defect_radius,
                                                    outer_radius,
                                                    (right - left));
    }
    else 
    {
        throw std::invalid_argument("Must input hypercube or hyperball to make_grid");
    }

    coarse_tria.copy_triangulation(tria);
    tria.refine_global(num_refines);

    if (refine_further_flag)
        refine_further();
}



template <int dim>
void NematicSystemMPIDriver<dim>::refine_further()
{
    dealii::Point<dim> center;
    double fine_left = left * 16.0 / 20.0;
    double fine_right = right * 16.0 / 20.0;
    for (auto &cell : tria.active_cell_iterators())
    {
        center = cell->center();
        if ((center[0] > fine_left) && (center[0] < fine_right)
            && (center[1] > fine_left) && (center[1] < fine_right))
            cell->set_refine_flag();
    }
    tria.execute_coarsening_and_refinement();

    fine_left = left * 8.0 / 20.0;
    fine_right = right * 8.0 / 20.0;
    for (auto &cell : tria.active_cell_iterators())
    {
        center = cell->center();
        if ((center[0] > fine_left) && (center[0] < fine_right)
            && (center[1] > fine_left) && (center[1] < fine_right))
            cell->set_refine_flag();
    }
    tria.execute_coarsening_and_refinement();

    fine_left = left * 4.0 / 20.0;
    fine_right = right * 4.0 / 20.0;
    for (auto &cell : tria.active_cell_iterators())
    {
        center = cell->center();
        if ((center[0] > fine_left) && (center[0] < fine_right)
            && (center[1] > fine_left) && (center[1] < fine_right))
            cell->set_refine_flag();
    }
    tria.execute_coarsening_and_refinement();
}



template <int dim>
void NematicSystemMPIDriver<dim>
::iterate_convex_splitting(NematicSystemMPI<dim> &nematic_system)
{
    unsigned int iterations = 0;
    double residual_norm{std::numeric_limits<double>::max()};
    while (residual_norm > simulation_tol && iterations < simulation_max_iters)
    {
        {
            dealii::TimerOutput::Scope t(computing_timer, "assembly");
            // nematic_system.assemble_system_anisotropic(dt);
            nematic_system.assemble_system(dt);
        }
        {
          dealii::TimerOutput::Scope t(computing_timer, "solve and update");
          nematic_system.solve_and_update(mpi_communicator, 
                                          simulation_newton_step);
        }
        residual_norm = nematic_system.return_norm();

        pcout << "Residual norm is: " << residual_norm << "\n";
        pcout << "Infinity norm is: " << nematic_system.return_linfty_norm() << "\n";

        iterations++;
    }

    if (residual_norm > simulation_tol)
        std::terminate();
}



template <int dim>
void NematicSystemMPIDriver<dim>::
iterate_forward_euler(NematicSystemMPI<dim> &nematic_system)
{
    {
        dealii::TimerOutput::Scope t(computing_timer, "assembly");
        nematic_system.assemble_system_forward_euler(dt);
    }
    {
        dealii::TimerOutput::Scope t(computing_timer, "solve and update");
        nematic_system.update_forward_euler(mpi_communicator);
    }
}



template <int dim>
void NematicSystemMPIDriver<dim>::
iterate_timestep(NematicSystemMPI<dim> &nematic_system)
{
    {
        dealii::TimerOutput::Scope t(computing_timer, "setup dofs");
        nematic_system.setup_dofs(mpi_communicator,
                                  /*initial_timestep = */ false,
                                  time_discretization);
    }

    if (time_discretization == std::string("convex_splitting"))
        iterate_convex_splitting(nematic_system);
    else if (time_discretization == std::string("forward_euler"))
        iterate_forward_euler(nematic_system);

//    nematic_system.assemble_rhs(dt);
//    nematic_system.solve_rhs(mpi_communicator);
    nematic_system.set_past_solution_to_current(mpi_communicator);
}



template <int dim>
void NematicSystemMPIDriver<dim>::run(std::string parameter_filename)
{
    dealii::ParameterHandler prm;
    std::ifstream ifs(parameter_filename);
    NematicSystemMPIDriver<dim>::declare_parameters(prm);
    NematicSystemMPI<dim>::declare_parameters(prm);
    prm.parse_input(ifs);
    get_parameters(prm);

    // prm.declare_entry(kGitHash, const std::string &default_value)

    make_grid();

    NematicSystemMPI<dim> nematic_system(tria, degree);
    nematic_system.get_parameters(prm);

    prm.print_parameters(data_folder + std::string("simulation_parameters.prm"));

    nematic_system.setup_dofs(mpi_communicator, true, time_discretization);
    {
        dealii::TimerOutput::Scope t(computing_timer, "initialize fe field");
        nematic_system.initialize_fe_field(mpi_communicator);
    }
    nematic_system.output_results(mpi_communicator, tria,
                                  data_folder, config_filename, 0);
    nematic_system.output_Q_components(mpi_communicator, tria,
                                       data_folder, 
                                       std::string("Q_components_") 
                                       + config_filename, 0);
//    nematic_system.assemble_rhs(dt);
//    nematic_system.solve_rhs(mpi_communicator);
//    nematic_system.output_rhs_components(mpi_communicator, tria, 
//                                         data_folder,
//                                         std::string("rhs_components_")
//                                         + config_filename, 0);

    for (unsigned int current_step = 1; current_step < n_steps; ++current_step)
    {
        pcout << "Starting timestep #" << current_step << "\n\n";

        iterate_timestep(nematic_system);
        {
            dealii::TimerOutput::Scope t(computing_timer, "find defects, calc energy");
            nematic_system.find_defects(defect_size, 
                                        defect_charge_threshold, 
                                        dt*current_step);
            nematic_system.calc_energy(mpi_communicator, dt*current_step);
        }

        if (current_step % vtu_interval == 0)
        {
            dealii::TimerOutput::Scope t(computing_timer, "output vtu");
            nematic_system.output_results(mpi_communicator, tria, data_folder,
                                          config_filename, current_step);
            nematic_system.output_Q_components(mpi_communicator, tria,
                                               data_folder, 
                                               std::string("Q_components_") 
                                               + config_filename, current_step);

//            nematic_system.output_rhs_components(mpi_communicator, tria, 
//                                                 data_folder,
//                                                 std::string("rhs_components_")
//                                                 + config_filename, current_step);
        }
        if (current_step % checkpoint_interval == 0)
        {
            dealii::TimerOutput::Scope t(computing_timer, "output checkpoint");
            try
            {
                nematic_system.output_defect_positions(mpi_communicator, 
                                                       data_folder, 
                                                       defect_filename);
                nematic_system.output_configuration_energies(mpi_communicator, 
                                                             data_folder, 
                                                             energy_filename);
                Serialization::serialize_nematic_system(mpi_communicator,
                                                        archive_filename,
                                                        degree,
                                                        coarse_tria,
                                                        tria,
                                                        nematic_system);
            }
            catch (std::exception &exc) 
            {
                std::cout << exc.what() << std::endl;
            }
        }

        pcout << "Finished timestep\n\n";
    }

}



template <int dim>
void NematicSystemMPIDriver<dim>::run_deserialization()
{
    std::string filename("nematic_simulation");

    std::unique_ptr<NematicSystemMPI<dim>> nematic_system
        = Serialization::deserialize_nematic_system(mpi_communicator,
                                                    filename,
                                                    degree,
                                                    coarse_tria,
                                                    tria,
                                                    time_discretization);

    nematic_system->output_results(mpi_communicator, tria, data_folder,
                                   config_filename, 0);
}

template class NematicSystemMPIDriver<2>;
template class NematicSystemMPIDriver<3>;
