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

template <int dim>
NematicSystemMPIDriver<dim>::
NematicSystemMPIDriver(unsigned int degree_,
                       unsigned int num_refines_,
                       double left_,
                       double right_,
                       double dt_,
                       unsigned int n_steps_,
                       double simulation_tol_,
                       unsigned int simulation_max_iters_,
                       std::string data_folder_,
                       std::string config_filename_,
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
    , left(left_)
    , right(right_)

    , dt(dt_)
    , n_steps(n_steps_)

    , simulation_tol(simulation_tol_)
    , simulation_max_iters(simulation_max_iters_)

    , data_folder(data_folder_)
    , config_filename(config_filename_)
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
    prm.declare_entry("Left",
                      "-1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Right",
                      "1.0",
                      dealii::Patterns::Double());

    prm.declare_entry("dt",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Number of steps",
                      "30",
                      dealii::Patterns::Integer());

    prm.declare_entry("Simulation tolerance",
                      "1e-10",
                      dealii::Patterns::Double());
    prm.declare_entry("Simulation maximum iterations",
                      "20",
                      dealii::Patterns::Integer());

    prm.declare_entry("Data folder",
                      "./",
                      dealii::Patterns::DirectoryName());
    prm.declare_entry("Configuration filename",
                      "nematic_configuration",
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
    left = prm.get_double("Left");
    right = prm.get_double("Right");

    dt = prm.get_double("dt");
    n_steps = prm.get_integer("Number of steps");

    simulation_tol = prm.get_double("Simulation tolerance");
    simulation_max_iters = prm.get_integer("Simulation maximum iterations");

    data_folder = prm.get("Data folder");
    config_filename = prm.get("Configuration filename");
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
    dealii::GridGenerator::hyper_cube(tria, left, right);
    coarse_tria.copy_triangulation(tria);
    tria.refine_global(num_refines);
}



template <int dim>
void NematicSystemMPIDriver<dim>::
iterate_timestep(NematicSystemMPI<dim> &nematic_system)
{
    {
        dealii::TimerOutput::Scope t(computing_timer, "setup dofs");
        nematic_system.setup_dofs(mpi_communicator,
                                  /*initial_timestep = */ false);
    }

    unsigned int iterations = 0;
    double residual_norm{std::numeric_limits<double>::max()};
    while (residual_norm > simulation_tol && iterations < simulation_max_iters)
    {
        {
            dealii::TimerOutput::Scope t(computing_timer, "assembly");
            nematic_system.assemble_system(dt);
        }
        {
          dealii::TimerOutput::Scope t(computing_timer, "solve and update");
          nematic_system.solve_and_update(mpi_communicator, 1.0);
        }
        residual_norm = nematic_system.return_norm();

        pcout << "Residual norm is: " << residual_norm << "\n";
    }

    if (residual_norm > simulation_tol)
        std::terminate();

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

    make_grid();

    NematicSystemMPI<dim> nematic_system(tria, degree);
    nematic_system.get_parameters(prm);

    nematic_system.setup_dofs(mpi_communicator, true);
    {
        dealii::TimerOutput::Scope t(computing_timer, "initialize fe field");
        nematic_system.initialize_fe_field(mpi_communicator);
    }
    nematic_system.output_results(mpi_communicator, tria,
                                  data_folder, config_filename, 0);
    for (int current_step = 1; current_step < n_steps; ++current_step)
    {
        pcout << "Starting timestep #" << current_step << "\n\n";

        iterate_timestep(nematic_system);
        {
            dealii::TimerOutput::Scope t(computing_timer, "output results");
            nematic_system.output_results(mpi_communicator, tria, data_folder,
                                          config_filename, current_step);
        }

        pcout << "Finished timestep\n\n";
    }

    serialize_nematic_system(nematic_system, archive_filename);
}



template <int dim>
void NematicSystemMPIDriver<dim>::run_deserialization()
{
    std::string filename("nematic_simulation");

    std::unique_ptr<NematicSystemMPI<dim>> nematic_system
        = deserialize_nematic_system(filename);

    nematic_system->output_results(mpi_communicator, tria, data_folder,
                                   config_filename, 0);
}



template <int dim>
void NematicSystemMPIDriver<dim>::
serialize_nematic_system(const NematicSystemMPI<dim> &nematic_system,
                         const std::string filename)
{
    const unsigned int my_id
        = dealii::Utilities::MPI::this_mpi_process (mpi_communicator);
    if (my_id == 0)
    {
        std::ofstream ofs(filename + std::string(".params.ar"));
        boost::archive::text_oarchive oa(ofs);

        oa << degree;
        oa << coarse_tria;
        oa << nematic_system;
    }

    dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
        sol_trans(nematic_system.return_dof_handler());
    sol_trans.
        prepare_for_serialization(nematic_system.return_current_solution());
    tria.save(filename + std::string(".mesh.ar"));
}



template <int dim>
std::unique_ptr<NematicSystemMPI<dim>> NematicSystemMPIDriver<dim>::
deserialize_nematic_system(const std::string filename)
{
    std::ifstream ifs(filename + std::string(".params.ar"));
    boost::archive::text_iarchive ia(ifs);

    ia >> degree;
    ia >> coarse_tria;
    tria.copy_triangulation(coarse_tria);

    std::unique_ptr<NematicSystemMPI<dim>> nematic_system
        = std::make_unique<NematicSystemMPI<dim>>(tria, degree);
    ia >> (*nematic_system);

    tria.load(filename + std::string(".mesh.ar"));
    nematic_system->setup_dofs(mpi_communicator, /*initial_step=*/true);

    const dealii::DoFHandler<dim>& dof_handler
        = nematic_system->return_dof_handler();
    const dealii::IndexSet locally_owned_dofs
        = dof_handler.locally_owned_dofs();
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
        sol_trans(dof_handler);
    sol_trans.deserialize(completely_distributed_solution);

    nematic_system->set_current_solution(mpi_communicator,
                                         completely_distributed_solution);
    nematic_system->set_past_solution_to_current(mpi_communicator);

    return std::move(nematic_system);
}

template class NematicSystemMPIDriver<2>;
template class NematicSystemMPIDriver<3>;