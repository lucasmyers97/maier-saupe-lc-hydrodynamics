#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

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
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/numerics/vector_tools_common.h>
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
                       unsigned int n_recentered_steps_,
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
    , n_recentered_steps(n_recentered_steps_)

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
                      "hypercube",
                      dealii::Patterns::Selection("hypercube"
                                                  "|hyperball"
                                                  "|two-defect-complement"),
                      "Type of grid to use for simulation");
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
    prm.declare_entry("Number of recentered steps",
                      "0",
                      dealii::Patterns::Integer(),
                      "Number of timesteps after grid is recentered around "
                      "defect");
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
    n_recentered_steps = prm.get_integer("Number of recentered steps");
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
void NematicSystemMPIDriver<dim>::
conditional_output(unsigned int timestep)
{
    if (timestep % vtu_interval == 0)
    {
        dealii::TimerOutput::Scope t(computing_timer, "output vtu");
        nematic_system->output_results(mpi_communicator, 
                                       tria,
                                       data_folder, 
                                       config_filename,
                                       timestep);
        nematic_system->output_Q_components(mpi_communicator, 
                                            tria,
                                            data_folder, 
                                            std::string("Q_components_") 
                                            + config_filename,
                                            timestep);
    }
    if (timestep % checkpoint_interval)
    {
        dealii::TimerOutput::Scope t(computing_timer, "output checkpoint");
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
        /** DIMENSIONALLY-DEPENDENT can be made to work in 3D */
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

    refine_further();
}



/** DIMENSIONALLY-DEPENDENT but can easily be made independent */
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
            if (grid_type == "hypercube")
                cell_distance = std::max(std::abs(grid_cell_difference[0]), 
                                         std::abs(grid_cell_difference[1]));
            else if (grid_type == "hyperball")
                cell_distance = grid_cell_difference.norm();

            if (cell_distance < refine_distance)
                cell->set_refine_flag();
        }

        tria.execute_coarsening_and_refinement();
    }
}



/** DIMENSIONALLY-DEPENDENT dependent on defects being points, 
 * could probably be made to be dimensionally-independent but it might be
 * better to just do a gradient-based adaptive refinement */
template <int dim>
void NematicSystemMPIDriver<dim>
::refine_around_defects()
{
    // should be initial defect points of configuration
    std::vector<dealii::Point<dim>> defect_pts;
    for (const auto &previous_defect_pt : previous_defect_points)
    {
        dealii::Point<dim> defect_pt;
        for (unsigned int i = 0; i < dim; ++i)
            defect_pt[i] = previous_defect_pt[i];

        defect_pts.push_back(defect_pt);
    }

    dealii::Point<dim> defect_cell_difference;
    double defect_cell_distance = 0;

    for (const auto &refine_dist : defect_refine_distances)
    {
        for (const auto &defect_pt : defect_pts)
            for (auto &cell : tria.active_cell_iterators())
            {
                if (!cell->is_locally_owned())
                    continue;

                defect_cell_difference = defect_pt - cell->center();
                defect_cell_distance = defect_cell_difference.norm();

                if (defect_cell_distance <= refine_dist)
                    cell->set_refine_flag();
            }

        tria.execute_coarsening_and_refinement();
    }
}



/** DIMENSIONALLY-DEPENDENT probably cannot be made independent */
template <int dim>
std::vector<std::size_t> NematicSystemMPIDriver<dim>
::sort_defect_points()
{
    if (defect_points.size() != previous_defect_points.size())
        throw std::length_error("defect_points and previous_defect_points "
                                "have different sizes");

    // find index of previous_defect_points which each defect_pt is closest to
    std::vector<std::size_t> defect_idx;
    for (const auto &defect_pt : defect_points)
    {
        std::vector<double> dist_to_previous_points(defect_points.size());
        for (std::size_t i = 0; i < defect_points.size(); ++i)
            for (std::size_t j = 0; j < dim; ++j)
                dist_to_previous_points[i] += 
                    (defect_pt[j] - previous_defect_points[i][j])
                    * (defect_pt[j] - previous_defect_points[i][j]);

        const auto min_dist = std::min_element(dist_to_previous_points.begin(),
                                               dist_to_previous_points.end());
        defect_idx.push_back(std::distance(dist_to_previous_points.begin(),
                                           min_dist));
    }

    // check whether there are duplicate indexes
    std::set<std::size_t> defect_idx_set(defect_idx.begin(), defect_idx.end());
    if (defect_idx_set.size() != defect_idx.size())
        throw std::runtime_error("Defect points do not uniquely correspond to "
                                 "previous defect points");

    return defect_idx;
}



/** DIMENSIONALLY-DEPENDENT probablly cannot be made independent */
template <int dim>
void NematicSystemMPIDriver<dim>
::recenter_defect_refinement()
{
    // need to link dof_handler to solution_transfer before coarsening/refining
    const dealii::DoFHandler<dim> &dof_handler 
        = nematic_system->return_dof_handler();
    dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
        soltrans(dof_handler);

    // turn defect_points into vector of dealii::Point's (easier to work with)
    std::vector<dealii::Point<dim>> defect_pts;
    for (const auto &defect_point : defect_points)
    {
        dealii::Point<dim> defect_pt;
        for (unsigned int i = 0; i < dim; ++i)
            defect_pt[i] = defect_point[i];
        defect_pts.push_back(defect_pt);
    }

    // set up necessary data about defect position and grid center
    dealii::Point<dim> cell_center;
    dealii::Point<dim> grid_center;
    dealii::Point<dim> grid_cell_difference;
    double grid_cell_distance = 0;
    double defect_cell_distance = 0;

    grid_center[0] = 0.5 * (left + right);
    grid_center[1] = grid_center[0];

    // each refine region is half the size of the previous
    std::vector<double> refine_distances(num_further_refines);
    for (std::size_t i = 0; i < num_further_refines; ++i)
        refine_distances[i] = std::pow(0.5, i + 1) * (right - grid_center[0]);
   
    // categorize each cell based on how many original further refinements
    // and how many new refinements, given that the defect has moved
    int num_original_refinements = 0;
    int num_further_refines = 0;
    int num_new_refinements = 0;
    for (auto &cell : tria.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        cell_center = cell->center();
        grid_cell_difference = grid_center - cell_center;
        
        // linfty norm for cube, l2norm for ball
        if (grid_type == "hypercube")
            grid_cell_distance = std::max(std::abs(grid_cell_difference[0]), 
                                          std::abs(grid_cell_difference[1]));
        else if (grid_type == "hyperball")
            grid_cell_distance = grid_cell_difference.norm();

        num_further_refines = 0;
        for (const auto refine_distance : refine_distances)
        {
            if (grid_cell_distance < refine_distance)
                    ++num_further_refines;
        }

        // get num_original_refinements
        num_original_refinements = cell->level()
                                   - num_refines - num_further_refines;

        // get num_new_refines
        num_new_refinements = 0;
        for (const auto &defect_refine_distance : defect_refine_distances)
            for (const auto &defect_pt : defect_pts)
            {
                defect_cell_distance = (defect_pt - cell_center).norm();
                if (defect_cell_distance < defect_refine_distance)
                {
                    ++num_new_refinements;
                    break;
                }
            }

        if ((num_original_refinements - num_new_refinements) == 1)
        {
            cell->set_coarsen_flag();
        }
        else if ((num_new_refinements - num_original_refinements) == 1)
        {
            cell->set_refine_flag();
        }
        else if ((num_original_refinements - num_new_refinements) == 0)
        {
            continue;
        }
        else
            throw std::runtime_error(
                    std::string("Too many coarsens or refines in "
                                "recenter_grid_refinement\n")
                    + std::string("num_new_refines is: ") 
                    + std::to_string(num_new_refinements)
                    + std::string("\n")
                    + std::string("num_old_refines is: ") 
                    + std::to_string(num_original_refinements)
                    + std::string("\n")
                    + std::string("cell level is: ") 
                    + std::to_string(cell->level())
                    + std::string("\n")
                    + std::string("non-defect cell level is: ")
                    + std::to_string(cell->level()
                                     - num_refines - num_further_refines)
                    + std::string("\n")
                    + std::string("x is: ") 
                    + std::string("\n")
                    + std::to_string(cell_center[0])
                    + std::string("\n")
                    + std::string("y is: ") 
                    + std::to_string(cell_center[1])
                    + std::string("\n"));
    }
    tria.prepare_coarsening_and_refinement();

    // interpolate solution on old grid to solution on new grid
    const LA::MPI::Vector &current_solution 
        = nematic_system->return_current_solution();
    soltrans.prepare_for_coarsening_and_refinement(current_solution);
    tria.execute_coarsening_and_refinement();

    // set up solution on new grid
    nematic_system->setup_dofs(mpi_communicator,
                              /* initial_timestep = */ true);
    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, 
                                                    locally_relevant_dofs);
    LA::MPI::Vector solution;
    solution.reinit(locally_owned_dofs,
                    mpi_communicator);
    soltrans.interpolate(solution);

    // transfer solution on new grid to nematic_system
    nematic_system->initialize_fe_field(mpi_communicator, solution);
}



template <int dim>
void NematicSystemMPIDriver<dim>::
iterate_timestep()
{
    {
        dealii::TimerOutput::Scope t(computing_timer, "setup dofs");
        nematic_system->setup_dofs(mpi_communicator,
                                   /*initial_timestep = */ false);
    }

    unsigned int iterations = 0;
    double residual_norm{std::numeric_limits<double>::max()};
    while (residual_norm > simulation_tol && iterations < simulation_max_iters)
    {
        {
            dealii::TimerOutput::Scope t(computing_timer, "assembly");
            nematic_system->assemble_system(dt, theta, time_discretization);
        }
        {
          dealii::TimerOutput::Scope t(computing_timer, "solve and update");
          nematic_system->solve_and_update(mpi_communicator, 
                                          simulation_newton_step);
        }
        residual_norm = nematic_system->return_norm();

        pcout << "Residual norm is: " << residual_norm << "\n";
        pcout << "Infinity norm is: " << nematic_system->return_linfty_norm() << "\n";

        iterations++;
    }

    if (residual_norm > simulation_tol)
        throw std::runtime_error("Newton iteration failed");

    nematic_system->set_past_solution_to_current(mpi_communicator);
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

    nematic_system = std::make_unique<NematicSystemMPI<dim>>(tria, degree);
    nematic_system->get_parameters(prm);

    prm.print_parameters(data_folder 
                         + std::string("simulation_parameters.prm"),
                         dealii::ParameterHandler::OutputStyle::
                         KeepDeclarationOrder);

    /** DIMENSIONALLY-DEPENDENT */
    for (const auto &defect_pt : nematic_system->return_initial_defect_pts())
    {
        std::vector<double> previous_defect_pt(dim);
        for (unsigned int i = 0; i < dim; ++i)
            previous_defect_pt[i] = defect_pt[i];

        previous_defect_points.push_back(previous_defect_pt);
    }

    make_grid();
    refine_around_defects(); /** DIMENSIONALLY-DEPENDENT */
    if (freeze_defects)
    {
        auto domain_defect_pts = nematic_system->return_initial_defect_pts();
        const std::size_t n_defects = domain_defect_pts.size();

        std::vector<dealii::types::material_id> defect_ids;
        for (std::size_t i = 1; i <= n_defects; ++i)
            defect_ids.push_back(i);

        SetDefectBoundaryConstraints::mark_defect_domains(tria, 
                                                          domain_defect_pts, 
                                                          defect_ids, 
                                                          defect_radius);
    }
    nematic_system->setup_dofs(mpi_communicator, true);
    {
        dealii::TimerOutput::Scope t(computing_timer, "initialize fe field");
        nematic_system->initialize_fe_field(mpi_communicator);
    }
    conditional_output(0);

    for (unsigned int current_step = 1; current_step < n_steps; ++current_step)
    {
        pcout << "Starting timestep #" << current_step << "\n\n";
        iterate_timestep();
        {
            dealii::TimerOutput::Scope t(computing_timer, "find defects, calc energy");
            /** DIMENSIONALLY-DEPENDENT */
            defect_points = dealii::Utilities::MPI::compute_set_union(
                    nematic_system->find_defects(defect_size, 
                                                 defect_charge_threshold, 
                                                 dt*current_step),
                    mpi_communicator
                    );

            /** DIMENSIONALLY-DEPENDENT probably cannot be made independent */
            if ((defect_points.size() == previous_defect_points.size())
                && (defect_points.size() > 0)
                && (defect_refine_distances.size() > 0))
            {
                std::vector<std::size_t> defects_idx = sort_defect_points();
                std::vector<double> defect_distances(defect_points.size());
                for (std::size_t i = 0; i < defect_points.size(); ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                        defect_distances[i] += 
                            (defect_points[i][j] 
                             - previous_defect_points[defects_idx[i]][j])
                            * (defect_points[i][j] 
                               - previous_defect_points[defects_idx[i]][j]);
                
                for (auto defect_dist : defect_distances)
                    defect_dist = std::sqrt(defect_dist);

                double max_defect_dist 
                    = *(std::max_element(defect_distances.begin(),
                                         defect_distances.end()));
                double min_refine_dist
                    = *(std::min_element(defect_refine_distances.begin(),
                                         defect_refine_distances.end()));

                if (max_defect_dist > (min_refine_dist / 2))
                {
                    recenter_defect_refinement();
                    previous_defect_points = defect_points;
                }
            }

            nematic_system->calc_energy(mpi_communicator, dt*current_step);
        }
        conditional_output(current_step);

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
