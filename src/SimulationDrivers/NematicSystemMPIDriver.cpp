#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

#include <deal.II/lac/generic_linear_algebra.h>
namespace LA = dealii::LinearAlgebraTrilinos;

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/hdf5.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>

#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/grid_refinement.h>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <stdexcept>
#include <string>
#include <limits>
#include <exception>
#include <fstream>
#include <iostream>
#include <utility>
#include <memory>

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Numerics/SetDefectBoundaryConstraints.hpp"
#include "Utilities/ParameterParser.hpp"
#include "Utilities/Serialization.hpp"
#include "Utilities/DefectGridGenerator.hpp"
#include "Utilities/maier_saupe_constants.hpp"
// #include "Utilities/git_version.hpp"

template <int dim>
NematicSystemMPIDriver<dim>::
NematicSystemMPIDriver(unsigned int degree_,
                       unsigned int num_refines_,
                       double left_,
                       double right_,
                       std::string grid_type_,
                       unsigned int num_further_refines_,
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
    , left(left_)
    , right(right_)
    , grid_type(grid_type_)
    , num_further_refines(num_further_refines_)

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
NematicSystemMPIDriver<dim>::
NematicSystemMPIDriver(std::unique_ptr<NematicSystemMPI<dim>> nematic_system,

                       const std::string& input_archive_filename,
                       const std::string& perturbation_archive_filename,
                       unsigned int starting_timestep,

                       unsigned int checkpoint_interval,
                       unsigned int vtu_interval,
                       const std::string& data_folder,
                       const std::string& archive_filename,
                       const std::string& config_filename,
                       const std::string& defect_filename,
                       const std::string& energy_filename,

                       double defect_charge_threshold,
                       double defect_size,

                       const std::string& grid_type,
                       const std::string& grid_arguments,
                       double left,
                       double right,
                       unsigned int num_refines,
                       unsigned int num_further_refines,
                       unsigned int max_grid_level,
                       unsigned int refine_interval,
                       double twist_angular_speed,
                       const std::string& defect_refine_axis,

                       const std::vector<double>& defect_refine_distances,

                       double defect_position,
                       double defect_radius,
                       double outer_radius,

                       unsigned int degree,
                       const std::string& time_discretization,
                       double theta,
                       double dt,
                       unsigned int n_steps,
                       double simulation_tol,
                       double simulation_newton_step,
                       unsigned int simulation_max_iters,
                       bool freeze_defects)
    : mpi_communicator(MPI_COMM_WORLD)
    , tria(mpi_communicator,
           typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening))

    , nematic_system(std::move(nematic_system))

    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      dealii::TimerOutput::summary,
                      dealii::TimerOutput::cpu_and_wall_times)

    , input_archive_filename(input_archive_filename)
    , perturbation_archive_filename(perturbation_archive_filename)
    , starting_timestep(starting_timestep)

    , checkpoint_interval(checkpoint_interval)
    , vtu_interval(vtu_interval)
    , data_folder(data_folder)
    , archive_filename(archive_filename)
    , config_filename(config_filename)
    , defect_filename(defect_filename)
    , energy_filename(energy_filename)

    , defect_charge_threshold(defect_charge_threshold)
    , defect_size(defect_size)

    , grid_type(grid_type)
    , grid_arguments(grid_arguments)
    , left(left)
    , right(right)
    , num_refines(num_refines)
    , num_further_refines(num_further_refines)
    , max_grid_level(max_grid_level)
    , refine_interval(refine_interval)
    , twist_angular_speed(twist_angular_speed)
    , defect_refine_axis(string_to_defect_refine_axis(defect_refine_axis))

    , defect_refine_distances(defect_refine_distances)

    , defect_position(defect_position)
    , defect_radius(defect_radius)
    , outer_radius(outer_radius)

    , degree(degree)
    , time_discretization(time_discretization)
    , theta(theta)
    , dt(dt)
    , n_steps(n_steps)
    , simulation_tol(simulation_tol)
    , simulation_newton_step(simulation_newton_step)
    , simulation_max_iters(simulation_max_iters)
    , freeze_defects(freeze_defects)
{}



template <int dim>
void NematicSystemMPIDriver<dim>::
declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("NematicSystemMPIDriver");

    prm.enter_subsection("File output");
    prm.declare_entry("Checkpoint interval",
                      "10",
                      dealii::Patterns::Integer(),
                      "Number of timesteps between archive writes");
    prm.declare_entry("Vtu interval",
                      "10",
                      dealii::Patterns::Integer(),
                      "Number of timesteps between visual output writes");
    prm.declare_entry("Data folder",
                      "./",
                      dealii::Patterns::DirectoryName(),
                      "Name of directory where data is written to; "
                      "Must end in /");
    prm.declare_entry("Archive filename",
                      "nematic_simulation",
                      dealii::Patterns::FileName(),
                      "Filename of archive (note: full path is necessary)");
    prm.declare_entry("Configuration filename",
                      "nematic_configuration",
                      dealii::Patterns::FileName(),
                      "Filename prefix of vtu outputs "
                      "(appended to data folder)");
    prm.declare_entry("Defect filename",
                      "defect_positions",
                      dealii::Patterns::FileName(),
                      "Filename of defect position data "
                      "(does not need .h5 suffix)");
    prm.declare_entry("Energy filename",
                      "configuration_energy",
                      dealii::Patterns::FileName(),
                      "Filename of configuration energy data "
                      "(does not need .h5 suffix)");
    prm.leave_subsection();

    prm.enter_subsection("Defect detection");
    prm.declare_entry("Defect size",
                      "2.0",
                      dealii::Patterns::Double(),
                      "Maximal distance the algorithm will look for minimum "
                      "S-values in search for defects");
    prm.declare_entry("Defect charge threshold",
                      "0.3",
                      dealii::Patterns::Double(),
                      "Charge threshold for minimum S-value to be defect");
    prm.leave_subsection();

    prm.enter_subsection("Grid");
    prm.declare_entry("Grid type",
                      "hyper_cube",
                      dealii::Patterns::Anything(),
                      "Type of grid to use for simulation");
    prm.declare_entry("Grid arguments",
                      "0.0, 1.0 : false",
                      dealii::Patterns::Anything(),
                      "Arguments as a string to pass into grid generator");
    prm.declare_entry("Left",
                      "-1.0",
                      dealii::Patterns::Double(),
                      "Left coordinate of hypercube. If using hyperball, "
                      "gives left extent of hyperball");
    prm.declare_entry("Right",
                      "1.0",
                      dealii::Patterns::Double(),
                      "Right coordinate of hypercube. If using hyperball, "
                      "gives right extend of hyperball");
    prm.declare_entry("Number of refines",
                      "6",
                      dealii::Patterns::Integer(),
                      "Number of global refines on the mesh");
    prm.declare_entry("Number of further refines",
                      "0",
                      dealii::Patterns::Integer(),
                      "Number of progressive refines a distance L * 1/2^n "
                      "from the center, where L is the distance from the "
                      "center to the edge, and n is the further refine "
                      " number. Lengths in L2 for hyperball, Linfinity for "
                      " hypercube");
    prm.declare_entry("Defect refine distances",
                      "",
                      dealii::Patterns::Anything(),
                      "Comma-separated list of distances from defects at which"
                      " further refines should happen.");
    prm.declare_entry("Defect position",
                      "20.0",
                      dealii::Patterns::Double(),
                      "Positions of defects for two-defect-complement grid");
    prm.declare_entry("Defect radius",
                      "2.5",
                      dealii::Patterns::Double(),
                      "Radius of defects for two-defect-complement grid");
    prm.declare_entry("Outer radius",
                      "5.0",
                      dealii::Patterns::Double(),
                      "Outer radius of hyperball part of "
                      "two-defect-complement grid");
    prm.leave_subsection();

    prm.enter_subsection("Simulation");
    prm.declare_entry("Finite element degree",
                      "1",
                      dealii::Patterns::Integer(),
                      "Degree of finite element used for Nematic on grid");
    prm.declare_entry("Time discretization",
                      "convex_splitting",
                      dealii::Patterns::Selection("convex_splitting"
                                                  "|forward_euler"
                                                  "|semi_implicit"),
                      "Type of time discretization");
    prm.declare_entry("Theta",
                      "0.0",
                      dealii::Patterns::Double(),
                      "Semi-implicit time discretization scheme parameter; "
                      "theta = 0 is fully implicit, theta = 1 is fully "
                      "explicit, theta = 1/2 is Crank-Nicolson");
    prm.declare_entry("dt",
                      "1.0",
                      dealii::Patterns::Double(),
                      "Discrete timestep length");
    prm.declare_entry("Number of steps",
                      "30",
                      dealii::Patterns::Integer(),
                      "Number of timesteps in simulation");
    prm.declare_entry("Simulation tolerance",
                      "1e-10",
                      dealii::Patterns::Double(),
                      "Maximal L2 norm of residual Newton scheme vector "
                      "before simulation progresses to next timestep");
    prm.declare_entry("Simulation newton step",
                      "1.0",
                      dealii::Patterns::Double(),
                      "Step size for each update to Newton's method");
    prm.declare_entry("Simulation maximum iterations",
                      "20",
                      dealii::Patterns::Integer(),
                      "Maximal iterations for simulation-level Newton's "
                      "method");
    prm.declare_entry("Freeze defects",
                      "false",
                      dealii::Patterns::Bool(),
                      "Whether to freeze defects in place with AffineConstraints");
    prm.leave_subsection();


    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPIDriver<dim>::
get_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("NematicSystemMPIDriver");

    prm.enter_subsection("File output");
    checkpoint_interval = prm.get_integer("Checkpoint interval");
    vtu_interval = prm.get_integer("Vtu interval");
    data_folder = prm.get("Data folder");
    archive_filename = prm.get("Archive filename");
    config_filename = prm.get("Configuration filename");
    defect_filename = prm.get("Defect filename");
    energy_filename = prm.get("Energy filename");
    prm.leave_subsection();

    prm.enter_subsection("Defect detection");
    defect_charge_threshold = prm.get_double("Defect charge threshold");
    defect_size = prm.get_double("Defect size");
    prm.leave_subsection();

    prm.enter_subsection("Grid");
    grid_type = prm.get("Grid type");
    grid_arguments = prm.get("Grid arguments");
    left = prm.get_double("Left");
    right = prm.get_double("Right");
    num_refines = prm.get_integer("Number of refines");
    num_further_refines = prm.get_integer("Number of further refines");

    const auto defect_refine_distances_str
        = ParameterParser::parse_delimited(prm.get("Defect refine distances"));
    for (const auto &defect_refine_dist : defect_refine_distances_str)
        defect_refine_distances.push_back(std::stod(defect_refine_dist));

    defect_position = prm.get_double("Defect position");
    defect_radius = prm.get_double("Defect radius");
    outer_radius = prm.get_double("Outer radius");
    prm.leave_subsection();

    prm.enter_subsection("Simulation");
    degree = prm.get_integer("Finite element degree");
    time_discretization = prm.get("Time discretization");
    theta = prm.get_double("Theta");
    dt = prm.get_double("dt");
    n_steps = prm.get_integer("Number of steps");
    simulation_tol = prm.get_double("Simulation tolerance");
    simulation_newton_step = prm.get_double("Simulation newton step");
    simulation_max_iters = prm.get_integer("Simulation maximum iterations");
    freeze_defects = prm.get_bool("Freeze defects");
    prm.leave_subsection();

    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPIDriver<dim>::
print_parameters(std::string filename, dealii::ParameterHandler &prm)
{
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        NematicSystemMPIDriver<dim>::declare_parameters(prm);
        NematicSystemMPI<dim>::declare_parameters(prm);
        
        dealii::ParameterHandler::OutputStyle style
            = dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder
              | dealii::ParameterHandler::OutputStyle::Description;

        prm.print_parameters(filename, style);
    }
}



template <int dim>
void NematicSystemMPIDriver<dim>::output_vtu(unsigned int timestep)
{
    dealii::TimerOutput::Scope t(computing_timer, "output vtu");
    nematic_system->output_Q_components(mpi_communicator, 
                                        tria,
                                        data_folder, 
                                        std::string("Q_components_") 
                                        + config_filename,
                                        timestep);
    pcout << "outputted\n";
    nematic_system->output_results(mpi_communicator, 
                                   tria,
                                   data_folder, 
                                   config_filename,
                                   timestep);
}



template <int dim>
void NematicSystemMPIDriver<dim>::output_checkpoint(unsigned int timestep)
{
    dealii::TimerOutput::Scope t(computing_timer, "output checkpoint");
    if (dim == 2)
        nematic_system->output_defect_positions(mpi_communicator, 
                                                data_folder, 
                                                defect_filename);
    
    nematic_system->output_configuration_energies(mpi_communicator, 
                                                  data_folder, 
                                                  energy_filename);
    Serialization::serialize_nematic_system(mpi_communicator,
                                            archive_filename
                                            + std::string("_")
                                            + std::to_string(timestep),
                                            degree,
                                            coarse_tria,
                                            tria,
                                            *nematic_system);
}



template <int dim>
void NematicSystemMPIDriver<dim>::
conditional_output(unsigned int timestep)
{
    if (timestep % vtu_interval == 0)
        output_vtu(timestep);
    if (timestep % checkpoint_interval == 0)
        output_checkpoint(timestep);
}



template <int dim>
void NematicSystemMPIDriver<dim>::make_grid()
{
    dealii::GridGenerator::generate_from_name_and_arguments(tria, 
                                                            grid_type, 
                                                            grid_arguments);

    coarse_tria.copy_triangulation(tria);
    tria.refine_global(num_refines);

    refine_further();
    refine_around_defects();
}



/** DIMENSIONALLY-WEIRD need more standardized grid refinement */
template <int dim>
void NematicSystemMPIDriver<dim>::refine_further()
{
    dealii::Point<dim> grid_center;
    dealii::Point<dim> cell_center;
    dealii::Point<dim> grid_cell_difference;
    double cell_distance = 0;

    grid_center[0] = 0.5 * (left + right);
    grid_center[1] = grid_center[0];

    // each refine region is half the size of the previous
    std::vector<double> refine_distances(num_further_refines);
    for (std::size_t i = 0; i < num_further_refines; ++i)
        refine_distances[i] = std::pow(0.5, i + 1) * (right - grid_center[0]);
    
    // refine each extra refinement zone
    for (const auto &refine_distance : refine_distances)
    {
        for (auto &cell : tria.active_cell_iterators())
        {
            cell_center = cell->center();
            grid_cell_difference = grid_center - cell_center;

            // linfty norm for cube, l2norm for ball
            if (grid_type == "hyper_cube")
                cell_distance = std::max(std::abs(grid_cell_difference[0]), 
                                         std::abs(grid_cell_difference[1]));
            else if (grid_type == "hyper_ball")
                cell_distance = grid_cell_difference.norm();
            else if (grid_type == "hyper_ball_balanced")
                cell_distance = std::max(std::abs(grid_cell_difference[0]), 
                                         std::abs(grid_cell_difference[1]));
            else
                continue;

            if (cell_distance < refine_distance)
                cell->set_refine_flag();
        }

        tria.execute_coarsening_and_refinement();
    }
}



/* NOTE: only works if disclinations are twisted around x-axis */
template <int dim>
void NematicSystemMPIDriver<dim>
::refine_around_defects()
{
    const std::vector<dealii::Point<dim>> &defect_pts 
        = nematic_system->return_initial_defect_pts();

    dealii::Point<dim> defect_cell_difference;
    dealii::Point<dim> twisted_defect_pt;
    dealii::Point<dim> center;
    double defect_cell_distance = 0;
    double rot_angle = 0;

    if (dim == 2 && defect_refine_axis != DefectRefineAxis::z)
        throw std::invalid_argument("In 2D, defects can only be refined along the z-axis");

    unsigned int i0 = 0;
    unsigned int i1 = 0;
    unsigned int i2 = 0;
    if (defect_refine_axis == DefectRefineAxis::x)
    {
        i0 = 0;
        i1 = 1;
        i2 = 2;
    } 
    else if (defect_refine_axis == DefectRefineAxis::y)
    {
        i0 = 1;
        i1 = 2;
        i2 = 0;
    }
    else if (defect_refine_axis == DefectRefineAxis::z)
    {
        i0 = 2;
        i1 = 0;
        i2 = 1;
    }
    else
    {
        throw std::invalid_argument("Incorrect defect refine axis encountered");
    }

    for (const auto &refine_dist : defect_refine_distances)
    {
        for (const auto &defect_pt : defect_pts)
            for (auto &cell : tria.active_cell_iterators())
            {
                if (!cell->is_locally_owned())
                    continue;

                rot_angle = dim == 3 ? twist_angular_speed * center[i0] : 0;

                center = cell->center();
                twisted_defect_pt[i1] = defect_pt[i1] * std::cos(rot_angle)
                                        - defect_pt[i2] * std::sin(rot_angle);
                twisted_defect_pt[i2] = defect_pt[i1] * std::sin(rot_angle)
                                        + defect_pt[i2] * std::cos(rot_angle);


                defect_cell_difference = twisted_defect_pt - center;
                defect_cell_distance = std::sqrt(defect_cell_difference[i1] * defect_cell_difference[i1]
                                                 + defect_cell_difference[i2] * defect_cell_difference[i2]);

                if (defect_cell_distance <= refine_dist)
                    cell->set_refine_flag();
            }

        tria.execute_coarsening_and_refinement();
    }
}



template <int dim>
void NematicSystemMPIDriver<dim>::setup_nematic_system()
{
    make_grid();
    
    if (freeze_defects)
        nematic_system->setup_dofs(mpi_communicator, tria, defect_radius);
    else
        nematic_system->setup_dofs(mpi_communicator, true);
    
    {
    dealii::TimerOutput::Scope t(computing_timer, "initialize fe field");
    nematic_system->initialize_fe_field(mpi_communicator);
    }
}



template <int dim>
void NematicSystemMPIDriver<dim>::setup_perturbed_nematic_system()
{
    dealii::GridGenerator::generate_from_name_and_arguments(tria, 
                                                            grid_type, 
                                                            grid_arguments);
    coarse_tria.copy_triangulation(tria);
    tria.load(perturbation_archive_filename + std::string(".mesh.ar"));

    if (freeze_defects)
        nematic_system->setup_dofs(mpi_communicator, tria, defect_radius);
    else
        nematic_system->setup_dofs(mpi_communicator, true);
    
    {
    dealii::TimerOutput::Scope t(computing_timer, "initialize fe field");
    nematic_system->initialize_fe_field(mpi_communicator);
    }

    dealii::FE_Q<dim> perturbation_fe(degree);
    dealii::DoFHandler<dim> perturbation_dof_handler(tria);
    perturbation_dof_handler.distribute_dofs(perturbation_fe);
    
    const dealii::IndexSet locally_owned_dofs 
        = perturbation_dof_handler.locally_owned_dofs();
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(perturbation_dof_handler,
                                                    locally_relevant_dofs);
    
    LA::MPI::Vector locally_relevant_perturbation(locally_owned_dofs,
                                                  locally_relevant_dofs,
                                                  mpi_communicator);
    
    {
        LA::MPI::Vector completely_distributed_perturbation(locally_owned_dofs,
                                                            mpi_communicator);
        dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
            sol_trans(perturbation_dof_handler);
    
        sol_trans.deserialize(completely_distributed_perturbation);
    
        locally_relevant_perturbation = completely_distributed_perturbation;
    }
    
    nematic_system->perturb_configuration_with_director(mpi_communicator,
                                                        perturbation_dof_handler,
                                                        locally_relevant_perturbation);
}



template <int dim>
void NematicSystemMPIDriver<dim>::setup_deserialized_nematic_system()
{
    std::ifstream ifs(input_archive_filename + std::string(".params.ar"));
    boost::archive::text_iarchive ia(ifs);
    
    ia >> degree;
    ia >> coarse_tria;
    
    dealii::GridGenerator::generate_from_name_and_arguments(tria, 
                                                            grid_type, 
                                                            grid_arguments);
    coarse_tria.copy_triangulation(tria);
    tria.load(input_archive_filename + std::string(".mesh.ar"));
    
    if (freeze_defects)
        nematic_system->setup_dofs(mpi_communicator, tria, defect_radius);
    else
        nematic_system->setup_dofs(mpi_communicator, true);
    
    const dealii::DoFHandler<dim>& dof_handler = nematic_system->return_dof_handler();
    const dealii::IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    LA::MPI::Vector completely_distributed_past_solution(locally_owned_dofs,
                                                         mpi_communicator);
    dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
        sol_trans(dof_handler);
    
    using vec = dealii::LinearAlgebraTrilinos::MPI::Vector;
    std::vector<vec*> serializing_vectors = { &completely_distributed_solution,
                                              &completely_distributed_past_solution };
    sol_trans.deserialize(serializing_vectors);
    
    nematic_system->set_current_solution(mpi_communicator,
                                         completely_distributed_solution);
    nematic_system->set_past_solution(mpi_communicator,
                                      completely_distributed_past_solution);
}



template <int dim>
void NematicSystemMPIDriver<dim>::refine_grid()
{
    dealii::Vector<float> estimated_error(tria.n_active_cells());

    nematic_system->setup_dofs(mpi_communicator, /*grid_modified = */ false);
    nematic_system->assemble_system(dt, theta, time_discretization);
    const auto& residual = nematic_system->return_residual();

    const auto& dof_handler = nematic_system->return_dof_handler();
    const auto& fe = dof_handler.get_fe();
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    // dealii::VectorTools::integrate_difference<dim, LA::MPI::Vector, dealii::Vector<float>>
    dealii::VectorTools::integrate_difference
        (dof_handler,
         residual,
         dealii::Functions::ZeroFunction<dim>(fe.n_components()),
         estimated_error,
         quadrature_formula,
         dealii::VectorTools::NormType::L2_norm);

    dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> soltrans(dof_handler);

    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error,
        0.3,
        0.03);

    if (tria.n_levels() > max_grid_level)
        for (auto &cell : tria.active_cell_iterators_on_level(max_grid_level))
            cell->clear_refine_flag();

    tria.prepare_coarsening_and_refinement();

    soltrans.prepare_for_coarsening_and_refinement(nematic_system->return_current_solution());
  
    tria.execute_coarsening_and_refinement();

    if (freeze_defects)
        nematic_system->setup_dofs(mpi_communicator, tria, defect_radius);
    else
        nematic_system->setup_dofs(mpi_communicator, true);

    const dealii::IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    soltrans.interpolate(completely_distributed_solution);

    nematic_system->set_current_solution(mpi_communicator,
                                         completely_distributed_solution);
    nematic_system->set_past_solution(mpi_communicator,
                                      completely_distributed_solution);
}



template <int dim>
void NematicSystemMPIDriver<dim>::refine_grid_from_disclination_charge()
{
    dealii::Vector<float> estimated_error(tria.n_active_cells());

    nematic_system->setup_dofs(mpi_communicator, /*grid_modified = */ false);
    nematic_system->assemble_system(dt, theta, time_discretization);
    const auto& residual = nematic_system->return_residual();

    const auto& dof_handler = nematic_system->return_dof_handler();
    const auto& fe = dof_handler.get_fe();
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    // dealii::VectorTools::integrate_difference<dim, LA::MPI::Vector, dealii::Vector<float>>
    dealii::VectorTools::integrate_difference
        (dof_handler,
         residual,
         dealii::Functions::ZeroFunction<dim>(fe.n_components()),
         estimated_error,
         quadrature_formula,
         dealii::VectorTools::NormType::L2_norm);

    dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> soltrans(dof_handler);

    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error,
        0.3,
        0.03);

    if (tria.n_levels() > max_grid_level)
        for (auto &cell : tria.active_cell_iterators_on_level(max_grid_level))
            cell->clear_refine_flag();

    tria.prepare_coarsening_and_refinement();

    soltrans.prepare_for_coarsening_and_refinement(nematic_system->return_current_solution());
  
    tria.execute_coarsening_and_refinement();

    if (freeze_defects)
        nematic_system->setup_dofs(mpi_communicator, tria, defect_radius);
    else
        nematic_system->setup_dofs(mpi_communicator, true);

    const dealii::IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    soltrans.interpolate(completely_distributed_solution);

    nematic_system->set_current_solution(mpi_communicator,
                                         completely_distributed_solution);
    nematic_system->set_past_solution(mpi_communicator,
                                      completely_distributed_solution);
}



template <int dim>
unsigned int NematicSystemMPIDriver<dim>::iterate_timestep()
{
    {
        dealii::TimerOutput::Scope t(computing_timer, "setup dofs");
        nematic_system->setup_dofs(mpi_communicator,
                                   /*initial_timestep = */ false);
    }

    double residual_norm{std::numeric_limits<double>::max()};
    for (unsigned int iterations = 0; iterations < simulation_max_iters; ++iterations)
    {
        if (time_discretization == "newtons_method")
            conditional_output(iterations);

        {
            dealii::TimerOutput::Scope t(computing_timer, "assembly");
            nematic_system->assemble_system(dt, theta, time_discretization);
            nematic_system->assemble_boundary_terms(dt, theta, time_discretization);
        }
        {
          dealii::TimerOutput::Scope t(computing_timer, "solve and update");
          nematic_system->solve_and_update(mpi_communicator, 
                                          simulation_newton_step);
        }
        residual_norm = nematic_system->return_norm();

        pcout << "Residual norm is: " << residual_norm << "\n";
        pcout << "Infinity norm is: " << nematic_system->return_linfty_norm() << "\n";

        if (residual_norm < simulation_tol)
        {
            nematic_system->set_past_solution_to_current(mpi_communicator);
            return iterations;
        }
    }

    throw std::runtime_error("Newton iteration failed");
}



template <int dim>
void NematicSystemMPIDriver<dim>::run()
{
    nematic_system->reinit_dof_handler(tria);

    if (input_archive_filename.empty() && perturbation_archive_filename.empty())
        setup_nematic_system();
    else if (input_archive_filename.empty() && !perturbation_archive_filename.empty())
        setup_perturbed_nematic_system();
    else
        setup_deserialized_nematic_system();

    if (time_discretization != "newtons_method")
    {
        nematic_system->calc_energy(mpi_communicator, 0, time_discretization);
        conditional_output(0);
    }

    pcout << "n_dofs is: " << nematic_system->return_dof_handler().n_dofs() << "\n\n";

    for (unsigned int current_step = starting_timestep; current_step <= n_steps; ++current_step)
    {
        pcout << "Starting timestep #" << current_step << "\n\n";
        unsigned int iterations = iterate_timestep();
        {
            dealii::TimerOutput::Scope t(computing_timer, "find defects, calc energy");
            if (dim == 2)
                nematic_system->find_defects(defect_size, 
                                             defect_charge_threshold, 
                                             dt*current_step);

            nematic_system->calc_energy(mpi_communicator, dt*current_step, time_discretization);
        }
        if (time_discretization != "newtons_method")
            conditional_output(current_step);
        else
        {
            output_vtu(iterations);
            output_checkpoint(iterations);
        }

        if (current_step % refine_interval == 0)
            refine_grid();

        computing_timer.print_summary();
        pcout << "Finished timestep\n\n";

    }
}



template <int dim>
std::unique_ptr<NematicSystemMPI<dim>> NematicSystemMPIDriver<dim>::
deserialize(const std::string &filename)
{
    std::unique_ptr<NematicSystemMPI<dim>> nematic_system
        = Serialization::deserialize_nematic_system(mpi_communicator,
                                                    filename,
                                                    degree,
                                                    coarse_tria,
                                                    tria,
                                                    time_discretization);

    return nematic_system;
}



template <int dim>
dealii::GridTools::Cache<dim> NematicSystemMPIDriver<dim>::get_grid_cache()
{
    return dealii::GridTools::Cache<dim>(tria);
}



template <int dim>
std::vector<dealii::BoundingBox<dim>> NematicSystemMPIDriver<dim>::
get_bounding_boxes(unsigned int refinement_level,
                   bool allow_merge,
                   unsigned int max_boxes)
{
    std::function<bool(const typename dealii::Triangulation<dim>::
                       active_cell_iterator &)>
        predicate_function = [](const typename dealii::Triangulation<dim>::
                                active_cell_iterator &cell)
        { return cell->is_locally_owned(); };

    return dealii::GridTools::
           compute_mesh_predicate_bounding_box(tria, 
                                               predicate_function,
                                               refinement_level,
                                               allow_merge,
                                               max_boxes);
}



template <int dim>
std::pair<std::vector<double>, std::vector<hsize_t>>
NematicSystemMPIDriver<dim>::
read_configuration_at_points(const NematicSystemMPI<dim> &nematic_system,
                             const std::vector<dealii::Point<dim>> &points,
                             const dealii::GridTools::Cache<dim> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<dim>>>
                             &global_bounding_boxes,
                             hsize_t offset)
{
    std::vector<typename dealii::Triangulation<dim>::active_cell_iterator>
        cells;
    std::vector<std::vector<dealii::Point<dim>>> qpoints;
    std::vector<std::vector<unsigned int>> maps;
    std::vector<std::vector<dealii::Point<dim>>> local_points;
    std::vector<std::vector<unsigned int>> owners;

    std::tie(cells, qpoints, maps, local_points, owners)
        = dealii::GridTools::
          distributed_compute_point_locations(cache,
                                              points, 
                                              global_bounding_boxes);

    // go through local cells and get values there
    std::vector<double> local_values;
    std::vector<hsize_t> local_value_indices;
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        if (!cells[i]->is_locally_owned())
            continue;

        std::size_t n_q = qpoints[i].size();

        dealii::Quadrature<dim> quad(qpoints[i]);
        dealii::FEValues<dim> 
            fe_values(nematic_system.return_dof_handler().get_fe(),
                      quad,
                      dealii::update_values);

        typename dealii::DoFHandler<dim>::active_cell_iterator 
            dof_cell(&nematic_system.return_dof_handler().get_triangulation(),
                     cells[i]->level(),
                     cells[i]->index(),
                     &nematic_system.return_dof_handler());
        fe_values.reinit(dof_cell);

        std::vector<dealii::Point<dim>> cell_points(n_q);
        std::vector<dealii::Vector<double>> 
            cell_values(n_q, dealii::Vector<double>(msc::vec_dim<dim>));
        for (std::size_t j = 0; j < n_q; ++j)
            cell_points[j] = local_points[i][j];

        fe_values.get_function_values(nematic_system.return_current_solution(),
                                      cell_values);

        for (std::size_t j = 0; j < n_q; ++j)
            for (std::size_t k = 0; k < msc::vec_dim<dim>; ++k)
            {
                local_values.push_back(cell_values[j][k]);
                // offset is in case we are writing a subset of a bigger list
                // of points
                local_value_indices.push_back(maps[i][j] + offset);
                local_value_indices.push_back(k);
            }
    }
    return std::make_pair(local_values, local_value_indices);
}

template class NematicSystemMPIDriver<2>;
template class NematicSystemMPIDriver<3>;
