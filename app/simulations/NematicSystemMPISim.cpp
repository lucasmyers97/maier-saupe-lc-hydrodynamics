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

        if (!tbl["nematic_system_mpi_driver"].is_table())
            throw std::invalid_argument("No nematic_system_mpi_driver table in toml file");
        const toml::table& nsmd_tbl = *tbl["nematic_system_mpi_driver"].as_table();

        const auto degree = nsmd_tbl["degree"].value<unsigned int>();
        if (!degree)
            throw std::invalid_argument("No nematic_system_mpi_driver.degree in toml file");

        if (!tbl["nematic_system_mpi"].is_table())
            throw std::invalid_argument("No nematic_system_mpi table in toml file");
        const toml::table& nsm_tbl = *tbl["nematic_system_mpi"].as_table();

        const auto field_theory = nsm_tbl["field_theory"]["field_theory"].value<std::string>();
        const auto L2 = nsm_tbl["field_theory"]["L2"].value<double>();
        const auto L3 = nsm_tbl["field_theory"]["L3"].value<double>();

        const auto maier_saupe_alpha = nsm_tbl["field_theory"]["maier_saupe"]["maier_saupe_alpha"].value<double>();
        const auto order = nsm_tbl["field_theory"]["maier_saupe"]["order"].value<int>();
        const auto lagrange_step_size = nsm_tbl["field_theory"]["maier_saupe"]["lagrange_step_size"].value<double>();
        const auto lagrange_tol = nsm_tbl["field_theory"]["maier_saupe"]["lagrange_tolerance"].value<double>();
        const auto lagrange_max_iter = nsm_tbl["field_theory"]["maier_saupe"]["lagrange_max_iter"].value<unsigned int>();

        if (!maier_saupe_alpha) throw std::invalid_argument("No maier_saupe_alpha in toml file");
        if (!order) throw std::invalid_argument("No order in toml file");
        if (!lagrange_step_size) throw std::invalid_argument("No lagrange_step_size in toml file");
        if (!lagrange_tol) throw std::invalid_argument("No lagrange_tol in toml file");
        if (!lagrange_max_iter) throw std::invalid_argument("No lagrange_max_iter in toml file");

        LagrangeMultiplierAnalytic<dim> lagrange_multiplier(order.value(), 
                                                            lagrange_step_size.value(), 
                                                            lagrange_tol.value(), 
                                                            lagrange_max_iter.value());

        const auto A = nsm_tbl["field_theory"]["landau_de_gennes"]["A"].value<double>();
        const auto B = nsm_tbl["field_theory"]["landau_de_gennes"]["B"].value<double>();
        const auto C = nsm_tbl["field_theory"]["landau_de_gennes"]["C"].value<double>();

        if (!field_theory) throw std::invalid_argument("No field_theory in toml file");
        if (!L2) throw std::invalid_argument("No L2 in toml file");
        if (!L3) throw std::invalid_argument("No L3 in toml file");
        if (!A) throw std::invalid_argument("No A in toml file");
        if (!B) throw std::invalid_argument("No B in toml file");
        if (!C) throw std::invalid_argument("No C in toml file");

        dealii::ParameterHandler prm;
        std::ifstream ifs(parameter_filename);
        NematicSystemMPIDriver<dim>::declare_parameters(prm);
        NematicSystemMPI<dim>::declare_parameters(prm);
        prm.parse_input(ifs);

        auto nematic_system 
            = std::make_unique<NematicSystemMPI<dim>>(degree.value(),
                                                      field_theory.value(),
                                                      L2.value(),
                                                      L3.value(),

                                                      maier_saupe_alpha.value(),

                                                      std::move(lagrange_multiplier),

                                                      A.value(),
                                                      B.value(),
                                                      C.value(),

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

                                                   degree.value(),
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
