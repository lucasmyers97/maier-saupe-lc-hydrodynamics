#include "BoundaryValues/DefectConfiguration.hpp"
#include "Utilities/ParameterParser.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "SimulationDrivers/NematicSystemMPIDriver.hpp"
#include "Parameters/toml.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/base/parameter_handler.h>

#include <boost/any.hpp>

#include <deal.II/base/types.h>
#include <string>
#include <map>

template <int dim>
std::unique_ptr<NematicSystemMPIDriver<dim>>
get_nematic_system_driver_from_paramters(const toml::table& tbl)
{
    if (!tbl["nematic_system_mpi"]["surface_potential_ids"].is_array())
        throw std::invalid_argument("No surface_potential_ids array in toml file");
    const auto surface_potential_ids
        = toml::convert<std::vector<dealii::types::boundary_id>>(
                    *tbl["nematic_system_mpi"]["surface_potential_ids"].as_array()
                    );

    if (!tbl["nematic_system_mpi"]["boundary_values"].is_array_of_tables())
        throw std::invalid_argument("No nematic_system_mpi.boundary_values array of tables in toml file");

    const toml::array& bv_array = *tbl["nematic_system_mpi"]["boundary_values"].as_array();
    std::map<dealii::types::boundary_id, std::unique_ptr<BoundaryValues<dim>>> boundary_value_funcs;
    for (const auto& bv_array_elem : bv_array)
    {
        const auto& bv_table = *bv_array_elem.as_table();
        const auto bv_params = BoundaryValuesFactory::parse_parameters<dim>(bv_table);
        const auto boundary_id = bv_table["boundary_id"].value<unsigned int>();
        if (!boundary_id) throw std::invalid_argument("No boundary_id in a boundary_value table");
        boundary_value_funcs[boundary_id.value()] = BoundaryValuesFactory::BoundaryValuesFactory<dim>(bv_params);
    }

    if (!tbl["nematic_system_mpi"]["initial_values"]["boundary_values"].is_table())
        throw std::invalid_argument("No nematic_system_mpi.initial_values table in toml file");
    const toml::table& in_tbl = *tbl["nematic_system_mpi"]["initial_values"]["boundary_values"].as_table();
    auto in_params = BoundaryValuesFactory::parse_parameters<dim>(in_tbl);


    if (!tbl["nematic_system_mpi"]["internal_boundary_values"]["left"]["boundary_values"].is_table())
        throw std::invalid_argument("No nematic_system_mpi.internal_boundary_values.left table in toml file");
    const toml::table& l_in_tbl = *tbl["nematic_system_mpi"]["internal_boundary_values"]["left"]["boundary_values"].as_table();
    auto l_in_params = BoundaryValuesFactory::parse_parameters<dim>(l_in_tbl);

    if (!tbl["nematic_system_mpi"]["internal_boundary_values"]["right"]["boundary_values"].is_table())
        throw std::invalid_argument("No nematic_system_mpi.internal_boundary_values.right table in toml file");
    const toml::table& r_in_tbl = *tbl["nematic_system_mpi"]["internal_boundary_values"]["right"]["boundary_values"].as_table();
    auto r_in_params = BoundaryValuesFactory::parse_parameters<dim>(r_in_tbl);


    auto initial_value_func = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(in_params);

    auto left_internal_boundary_func = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(l_in_params);

    auto right_internal_boundary_func = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(r_in_params);

    if (!tbl["nematic_system_mpi_driver"].is_table())
        throw std::invalid_argument("No nematic_system_mpi_driver table in toml file");
    const toml::table& nsmd_tbl = *tbl["nematic_system_mpi_driver"].as_table();

    const auto degree = nsmd_tbl["simulation"]["finite_element_degree"].value<unsigned int>();
    if (!degree)
        throw std::invalid_argument("No nematic_system_mpi_driver.degree in toml file");

    if (!tbl["nematic_system_mpi"].is_table())
        throw std::invalid_argument("No nematic_system_mpi table in toml file");
    const toml::table& nsm_tbl = *tbl["nematic_system_mpi"].as_table();

    const auto field_theory = nsm_tbl["field_theory"]["field_theory"].value<std::string>();
    const auto L2 = nsm_tbl["field_theory"]["L2"].value<double>();
    const auto L3 = nsm_tbl["field_theory"]["L3"].value<double>();

    const auto maier_saupe_alpha = nsm_tbl["field_theory"]["maier_saupe"]["maier_saupe_alpha"].value<double>();
    const auto order = nsm_tbl["field_theory"]["maier_saupe"]["lebedev_order"].value<int>();
    const auto lagrange_step_size = nsm_tbl["field_theory"]["maier_saupe"]["lagrange_step_size"].value<double>();
    const auto lagrange_tol = nsm_tbl["field_theory"]["maier_saupe"]["lagrange_tolerance"].value<double>();
    const auto lagrange_max_iter = nsm_tbl["field_theory"]["maier_saupe"]["lagrange_maximum_iterations"].value<unsigned int>();

    if (!maier_saupe_alpha) throw std::invalid_argument("No maier_saupe_alpha in toml file");
    if (!order) throw std::invalid_argument("No order in toml file");
    if (!lagrange_step_size) throw std::invalid_argument("No lagrange_step_size in toml file");
    if (!lagrange_tol) throw std::invalid_argument("No lagrange_tol in toml file");
    if (!lagrange_max_iter) throw std::invalid_argument("No lagrange_max_iter in toml file");

    LagrangeMultiplierAnalytic<dim> lagrange_multiplier(order.value(), 
                                                        lagrange_step_size.value(), 
                                                        lagrange_tol.value(), 
                                                        lagrange_max_iter.value());

    const auto S0 = nsm_tbl["field_theory"]["S0"].value<double>();
    const auto W1 = nsm_tbl["field_theory"]["W1"].value<double>();
    const auto W2 = nsm_tbl["field_theory"]["W2"].value<double>();
    const auto omega = nsm_tbl["field_theory"]["omega"].value<double>();

    if (!S0) throw std::invalid_argument("No S0 in toml file");
    if (!W1) throw std::invalid_argument("No W1 in toml file");
    if (!W2) throw std::invalid_argument("No W2 in toml file");
    if (!omega) throw std::invalid_argument("No omega in toml file");

    const auto A = nsm_tbl["field_theory"]["landau_de_gennes"]["A"].value<double>();
    const auto B = nsm_tbl["field_theory"]["landau_de_gennes"]["B"].value<double>();
    const auto C = nsm_tbl["field_theory"]["landau_de_gennes"]["C"].value<double>();

    if (!field_theory) throw std::invalid_argument("No field_theory in toml file");
    if (!L2) throw std::invalid_argument("No L2 in toml file");
    if (!L3) throw std::invalid_argument("No L3 in toml file");
    if (!A) throw std::invalid_argument("No A in toml file");
    if (!B) throw std::invalid_argument("No B in toml file");
    if (!C) throw std::invalid_argument("No C in toml file");

    auto nematic_system 
        = std::make_unique<NematicSystemMPI<dim>>(degree.value(),
                                                  field_theory.value(),
                                                  L2.value(),
                                                  L3.value(),

                                                  maier_saupe_alpha.value(),

                                                  S0.value(),
                                                  W1.value(),
                                                  W2.value(),
                                                  omega.value(),

                                                  std::move(lagrange_multiplier),

                                                  A.value(),
                                                  B.value(),
                                                  C.value(),

                                                  std::move(boundary_value_funcs),
                                                  std::move(initial_value_func),
                                                  std::move(left_internal_boundary_func),
                                                  std::move(right_internal_boundary_func),
                                                  std::move(surface_potential_ids));

    const auto input_archive_filename = nsmd_tbl["input_archive_filename"].value<std::string>();
    const auto perturbation_archive_filename = nsmd_tbl["perturbation_archive_filename"].value<std::string>();
    const auto starting_timestep = nsmd_tbl["starting_timestep"].value<unsigned int>();

    const auto checkpoint_interval = nsmd_tbl["file_output"]["checkpoint_interval"].value<unsigned int>();
    const auto vtu_interval = nsmd_tbl["file_output"]["vtu_interval"].value<unsigned int>();
    const auto data_folder = nsmd_tbl["file_output"]["data_folder"].value<std::string>();
    const auto archive_filename = nsmd_tbl["file_output"]["archive_filename"].value<std::string>();
    const auto configuration_filename = nsmd_tbl["file_output"]["configuration_filename"].value<std::string>();
    const auto defect_filename = nsmd_tbl["file_output"]["defect_filename"].value<std::string>();
    const auto energy_filename = nsmd_tbl["file_output"]["energy_filename"].value<std::string>();

    const auto defect_charge_threshold = nsmd_tbl["defect_detection"]["defect_charge_threshold"].value<double>();
    const auto defect_size = nsmd_tbl["defect_detection"]["defect_size"].value<double>();

    const auto grid_type = nsmd_tbl["grid"]["grid_type"].value<std::string>();
    const auto grid_arguments = nsmd_tbl["grid"]["grid_arguments"].value<std::string>();
    const auto left = nsmd_tbl["grid"]["left"].value<double>();
    const auto right = nsmd_tbl["grid"]["right"].value<double>();
    const auto number_of_refines = nsmd_tbl["grid"]["number_of_refines"].value<unsigned int>();
    const auto number_of_further_refines = nsmd_tbl["grid"]["number_of_further_refines"].value<unsigned int>();
    const auto max_grid_level = nsmd_tbl["grid"]["max_grid_level"].value<unsigned int>();
    const auto refine_interval = nsmd_tbl["grid"]["refine_interval"].value<unsigned int>();
    const auto twist_angular_speed = nsmd_tbl["grid"]["twist_angular_speed"].value<double>();
    const auto defect_refine_axis = nsmd_tbl["grid"]["defect_refine_axis"].value<std::string>();

    if (!nsmd_tbl["grid"]["defect_refine_distances"].is_array())
        throw std::invalid_argument("No defect_refine_distances array in toml file");
    const auto defect_refine_distances
        = toml::convert<std::vector<double>>(
                    *nsmd_tbl["grid"]["defect_refine_distances"].as_array()
                    );

    const auto defect_position = nsmd_tbl["grid"]["defect_position"].value<double>();
    const auto defect_radius = nsmd_tbl["grid"]["defect_radius"].value<double>();
    const auto outer_radius = nsmd_tbl["grid"]["outer_radius"].value<double>();

    const auto time_discretization = nsmd_tbl["simulation"]["time_discretization"].value<std::string>();
    const auto theta = nsmd_tbl["simulation"]["theta"].value<double>();
    const auto dt = nsmd_tbl["simulation"]["dt"].value<double>();
    const auto number_of_steps = nsmd_tbl["simulation"]["number_of_steps"].value<unsigned int>();
    const auto simulation_tolerance = nsmd_tbl["simulation"]["simulation_tolerance"].value<double>();
    const auto simulation_newton_step = nsmd_tbl["simulation"]["simulation_newton_step"].value<double>();
    const auto simulation_maximum_iterations = nsmd_tbl["simulation"]["simulation_maximum_iterations"].value<unsigned int>();
    const auto freeze_defects = nsmd_tbl["simulation"]["freeze_defects"].value<bool>();

    if (!input_archive_filename) throw std::invalid_argument("No input_archive_filename in toml file");
    if (!perturbation_archive_filename) throw std::invalid_argument("No perturbation_archive_filename in toml file");
    if (!starting_timestep) throw std::invalid_argument("No starting_timestep in toml file");

    if (!checkpoint_interval) throw std::invalid_argument("No checkpoint_interval in toml file");
    if (!vtu_interval) throw std::invalid_argument("No vtu_interval in toml file");
    if (!data_folder) throw std::invalid_argument("No data_folder in toml file");
    if (!archive_filename) throw std::invalid_argument("No archive_filename in toml file");
    if (!configuration_filename) throw std::invalid_argument("No configuration_filename in toml file");
    if (!defect_filename) throw std::invalid_argument("No defect_filename in toml file");
    if (!energy_filename) throw std::invalid_argument("No energy_filename in toml file");

    if (!defect_charge_threshold) throw std::invalid_argument("No defect_charge_threshold in toml file");
    if (!defect_size) throw std::invalid_argument("No defect_size in toml file");

    if (!grid_type) throw std::invalid_argument("No grid_type in toml file");
    if (!grid_arguments) throw std::invalid_argument("No grid_arguments in toml file");
    if (!left) throw std::invalid_argument("No left in toml file");
    if (!right) throw std::invalid_argument("No right in toml file");
    if (!number_of_refines) throw std::invalid_argument("No number_of_refines in toml file");
    if (!number_of_further_refines) throw std::invalid_argument("No number_of_further_refines in toml file");
    if (!max_grid_level) throw std::invalid_argument("No max_grid_level in toml file");
    if (!refine_interval) throw std::invalid_argument("No refine_interval in toml file");
    if (!twist_angular_speed) throw std::invalid_argument("No twist_angular_speed in toml file");
    if (!defect_refine_axis) throw std::invalid_argument("No defect_refine_axis in toml file");

    if (!defect_position) throw std::invalid_argument("No defect_position in toml file");
    if (!defect_radius) throw std::invalid_argument("No defect_radius in toml file");
    if (!outer_radius) throw std::invalid_argument("No outer_radius in toml file");

    if (!time_discretization) throw std::invalid_argument("No time_discretization in toml file");
    if (!theta) throw std::invalid_argument("No theta in toml file");
    if (!dt) throw std::invalid_argument("No dt in toml file");
    if (!number_of_steps) throw std::invalid_argument("No number_of_steps in toml file");
    if (!simulation_tolerance) throw std::invalid_argument("No simulation_tolerance in toml file");
    if (!simulation_newton_step) throw std::invalid_argument("No simulation_newton_step in toml file");
    if (!simulation_maximum_iterations) throw std::invalid_argument("No simulation_maximum_iterations in toml file");
    if (!freeze_defects) throw std::invalid_argument("No freeze_defects in toml file");

    auto nematic_driver
        = std::make_unique<NematicSystemMPIDriver<dim>>(std::move(nematic_system),

                                                        input_archive_filename.value(),
                                                        perturbation_archive_filename.value(),
                                                        starting_timestep.value(),

                                                        checkpoint_interval.value(),
                                                        vtu_interval.value(),
                                                        data_folder.value(),
                                                        archive_filename.value(),
                                                        configuration_filename.value(),
                                                        defect_filename.value(),
                                                        energy_filename.value(),

                                                        defect_charge_threshold.value(),
                                                        defect_size.value(),

                                                        grid_type.value(),
                                                        grid_arguments.value(),
                                                        left.value(),
                                                        right.value(),
                                                        number_of_refines.value(),
                                                        number_of_further_refines.value(),
                                                        max_grid_level.value(),
                                                        refine_interval.value(),
                                                        twist_angular_speed.value(),
                                                        defect_refine_axis.value(),

                                                        defect_refine_distances,

                                                        defect_position.value(),
                                                        defect_radius.value(),
                                                        outer_radius.value(),

                                                        degree.value(),
                                                        time_discretization.value(),
                                                        theta.value(),
                                                        dt.value(),
                                                        number_of_steps.value(),
                                                        simulation_tolerance.value(),
                                                        simulation_newton_step.value(),
                                                        simulation_maximum_iterations.value(),
                                                        freeze_defects.value());

    return nematic_driver;
}

int main(int ac, char* av[])
{
    try
    {
        if (ac - 1 != 1)
            throw std::invalid_argument("Error! Didn't input filename");
        std::string toml_filename(av[1]);

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

        const toml::table tbl = toml::parse_file(toml_filename);

        const auto dim = tbl["dim"].value<int>();
        if (!dim) throw std::invalid_argument("No dim in toml file");

        if (dim.value() == 2)
        {
            auto nematic_driver = get_nematic_system_driver_from_paramters<2>(tbl);
            nematic_driver->run();
        } else if (dim.value() == 3)
        {
            auto nematic_driver = get_nematic_system_driver_from_paramters<3>(tbl);
            nematic_driver->run();
        } else
            throw std::invalid_argument("dim argument in toml file must be 2 or 3");

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
