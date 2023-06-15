#include "Utilities/ParameterParser.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "SimulationDrivers/NematicSystemMPIDriver.hpp"
#include "Parameters/toml.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/base/parameter_handler.h>

#include <boost/any.hpp>

#include <string>
#include <map>

int main(int ac, char* av[])
{
    try
    {
        if (ac - 1 != 2)
            throw std::invalid_argument("Error! Didn't input two filenames");
        std::string parameter_filename(av[1]);
        std::string toml_filename(av[2]);

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

        const int dim = 2;

        const toml::table tbl = toml::parse_file(toml_filename);

        if (!tbl["nematic_system_mpi"].is_table())
            throw std::invalid_argument("No nematic_system_mpi table in toml file");
        const toml::table& bv_tbl = *tbl["nematic_system_mpi"].as_table();
        auto bv_params = BoundaryValuesFactory::parse_parameters<dim>(bv_tbl);

        if (!tbl["nematic_system_mpi"]["initial_values"].is_table())
            throw std::invalid_argument("No nematic_system_mpi.initial_values table in toml file");
        const toml::table& in_tbl = *tbl["nematic_system_mpi"]["initial_values"].as_table();
        auto in_params = BoundaryValuesFactory::parse_parameters<dim>(in_tbl);

        if (!tbl["nematic_system_mpi"]["internal_boundary_values"]["left"].is_table())
            throw std::invalid_argument("No nematic_system_mpi.internal_boundary_values.left table in toml file");
        const toml::table& l_in_tbl = *tbl["nematic_system_mpi"]["internal_boundary_values"]["left"].as_table();
        auto l_in_params = BoundaryValuesFactory::parse_parameters<dim>(l_in_tbl);

        if (!tbl["nematic_system_mpi"]["internal_boundary_values"]["right"].is_table())
            throw std::invalid_argument("No nematic_system_mpi.internal_boundary_values.right table in toml file");
        const toml::table& r_in_tbl = *tbl["nematic_system_mpi"]["internal_boundary_values"]["right"].as_table();
        auto r_in_params = BoundaryValuesFactory::parse_parameters<dim>(r_in_tbl);

        auto boundary_value_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(bv_params);

        auto initial_value_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(in_params);

        auto left_internal_boundary_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(l_in_params);

        auto right_internal_boundary_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(r_in_params);

        dealii::ParameterHandler prm;
        std::ifstream ifs(parameter_filename);
        NematicSystemMPIDriver<dim>::declare_parameters(prm);
        NematicSystemMPI<dim>::declare_parameters(prm);
        prm.parse_input(ifs);
        
        prm.enter_subsection("NematicSystemMPIDriver");
        prm.enter_subsection("Simulation");
        unsigned int degree = prm.get_integer("Finite element degree");
        prm.leave_subsection();
        prm.leave_subsection();

        prm.enter_subsection("Nematic system MPI");

        prm.enter_subsection("Field theory");
        std::string field_theory = prm.get("Field theory");
        double L2 = prm.get_double("L2");
        double L3 = prm.get_double("L3");

        prm.enter_subsection("Maier saupe");
        double maier_saupe_alpha = prm.get_double("Maier saupe alpha");

        int order = prm.get_integer("Lebedev order");
        double lagrange_step_size = prm.get_double("Lagrange step size");
        double lagrange_tol = prm.get_double("Lagrange tolerance");
        int lagrange_max_iter = prm.get_integer("Lagrange maximum iterations");

        LagrangeMultiplierAnalytic<dim> lagrange_multiplier(order, 
                                                            lagrange_step_size, 
                                                            lagrange_tol, 
                                                            lagrange_max_iter);
        prm.leave_subsection();

        prm.enter_subsection("Landau-de gennes");
        double A = prm.get_double("A");
        double B = prm.get_double("B");
        double C = prm.get_double("C");
        prm.leave_subsection();

        prm.leave_subsection();

        prm.leave_subsection();

        auto nematic_system 
            = std::make_unique<NematicSystemMPI<dim>>(degree,
                                                      field_theory,
                                                      L2,
                                                      L3,

                                                      maier_saupe_alpha,

                                                      std::move(lagrange_multiplier),

                                                      A,
                                                      B,
                                                      C,

                                                      std::move(boundary_value_func),
                                                      std::move(initial_value_func),
                                                      std::move(left_internal_boundary_func),
                                                      std::move(right_internal_boundary_func));

        prm.enter_subsection("NematicSystemMPIDriver");

        prm.enter_subsection("File output");
        unsigned int checkpoint_interval = prm.get_integer("Checkpoint interval");
        unsigned int vtu_interval = prm.get_integer("Vtu interval");
        std::string data_folder = prm.get("Data folder");
        std::string archive_filename = prm.get("Archive filename");
        std::string config_filename = prm.get("Configuration filename");
        std::string defect_filename = prm.get("Defect filename");
        std::string energy_filename = prm.get("Energy filename");
        prm.leave_subsection();

        prm.enter_subsection("Defect detection");
        double defect_charge_threshold = prm.get_double("Defect charge threshold");
        double defect_size = prm.get_double("Defect size");
        prm.leave_subsection();

        prm.enter_subsection("Grid");
        std::string grid_type = prm.get("Grid type");
        std::string grid_arguments = prm.get("Grid arguments");
        double left = prm.get_double("Left");
        double right = prm.get_double("Right");
        unsigned int num_refines = prm.get_integer("Number of refines");
        unsigned int num_further_refines = prm.get_integer("Number of further refines");

        std::vector<double> defect_refine_distances;
        const auto defect_refine_distances_str
            = ParameterParser::parse_delimited(prm.get("Defect refine distances"));
        for (const auto &defect_refine_dist : defect_refine_distances_str)
            defect_refine_distances.push_back(std::stod(defect_refine_dist));

        double defect_position = prm.get_double("Defect position");
        double defect_radius = prm.get_double("Defect radius");
        double outer_radius = prm.get_double("Outer radius");
        prm.leave_subsection();

        prm.enter_subsection("Simulation");
        std::string time_discretization = prm.get("Time discretization");
        double theta = prm.get_double("Theta");
        double dt = prm.get_double("dt");
        unsigned int n_steps = prm.get_integer("Number of steps");
        double simulation_tol = prm.get_double("Simulation tolerance");
        double simulation_newton_step = prm.get_double("Simulation newton step");
        unsigned int simulation_max_iters = prm.get_integer("Simulation maximum iterations");
        bool freeze_defects = prm.get_bool("Freeze defects");
        prm.leave_subsection();

        prm.leave_subsection();

        NematicSystemMPIDriver<dim> nematic_driver(std::move(nematic_system),
                                                   checkpoint_interval,
                                                   vtu_interval,
                                                   data_folder,
                                                   archive_filename,
                                                   config_filename,
                                                   defect_filename,
                                                   energy_filename,

                                                   defect_charge_threshold,
                                                   defect_size,

                                                   grid_type,
                                                   grid_arguments,
                                                   left,
                                                   right,
                                                   num_refines,
                                                   num_further_refines,

                                                   defect_refine_distances,

                                                   defect_position,
                                                   defect_radius,
                                                   outer_radius,

                                                   degree,
                                                   time_discretization,
                                                   theta,
                                                   dt,
                                                   n_steps,
                                                   simulation_tol,
                                                   simulation_newton_step,
                                                   simulation_max_iters,
                                                   freeze_defects);
        nematic_driver.run();

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Got exception which wasn't caught" << std::endl;
        return -1;
    }
}
