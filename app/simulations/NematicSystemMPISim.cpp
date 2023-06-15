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
        auto am = BoundaryValuesFactory::parse_parameters<dim>(bv_tbl);

        std::cout << boost::any_cast<std::string>(am["boundary-values-name"]) << "\n";
        std::cout << boost::any_cast<std::string>(am["boundary-condition"]) << "\n";
        std::cout << boost::any_cast<double>(am["S-value"]) << "\n";

        for (const auto &p : boost::any_cast<std::vector<std::vector<double>>>(am["defect-positions"]))
        {
            for (auto q : p)
                std::cout << q << " ";
            std::cout << "\n";
        }
        for (const double c : boost::any_cast<std::vector<double>>(am["defect-charges"]))
            std::cout << c << "\n";
        for (const double p : boost::any_cast<std::vector<double>>(am["defect-orientations"]))
            std::cout << p << "\n";

        std::cout << boost::any_cast<double>(am["defect-radius"]) << "\n";
        std::cout << boost::any_cast<std::string>(am["defect-charge-name"]) << "\n";

        std::cout << boost::any_cast<double>(am["anisotropy-eps"]) << "\n"; 
        std::cout << boost::any_cast<unsigned int>(am["degree"]) << "\n"; 
        std::cout << boost::any_cast<double>(am["charge"]) << "\n"; 
        std::cout << boost::any_cast<unsigned int>(am["n-refines"]) << "\n"; 
        std::cout << boost::any_cast<double>(am["tol"]) << "\n"; 
        std::cout << boost::any_cast<unsigned int>(am["max-iter"]) << "\n"; 
        std::cout << boost::any_cast<double>(am["newton-step"]) << "\n"; 

        std::cout << boost::any_cast<double>(am["phi"]) << "\n"; 
        std::cout << boost::any_cast<double>(am["k"]) << "\n"; 
        std::cout << boost::any_cast<double>(am["eps"]) << "\n"; 

        std::cout << boost::any_cast<double>(am["defect-distance"]) << "\n";
        std::cout << boost::any_cast<std::string>(am["defect-position-name"]) << "\n";
        std::cout << boost::any_cast<std::string>(am["defect-isomorph-name"]) << "\n";

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

        std::map<std::string, boost::any> bv_params;

        prm.enter_subsection("Boundary values");
        bv_params["boundary-values-name"] = prm.get("Name");
        bv_params["boundary-condition"] = prm.get("Boundary condition");
        bv_params["S-value"] = prm.get_double("S value");

        prm.enter_subsection("Defect configurations");
        bv_params["defect-positions"] 
            = ParameterParser::
              parse_coordinate_list<dim>(prm.get("Defect positions"));
        bv_params["defect-charges"]
            = ParameterParser::
              parse_number_list(prm.get("Defect charges"));
        bv_params["defect-orientations"]
            = ParameterParser::
              parse_number_list(prm.get("Defect orientations"));
        bv_params["defect-radius"] = prm.get_double("Defect radius");
        bv_params["defect-charge-name"] = prm.get("Defect charge name");
        prm.leave_subsection();

        prm.enter_subsection("Dzyaloshinskii");
        bv_params["anisotropy-eps"] = prm.get_double("Anisotropy eps");
        bv_params["degree"] = prm.get_integer("Degree");
        bv_params["charge"] = prm.get_double("Charge");
        bv_params["n-refines"] = prm.get_integer("N refines");
        bv_params["tol"] = prm.get_double("Tol");
        bv_params["max-iter"] = prm.get_integer("Max iter");
        bv_params["newton-step"] = prm.get_double("Newton step");
        prm.leave_subsection();

        prm.enter_subsection("Periodic configurations");
        bv_params["phi"] = prm.get_double("Phi");
        bv_params["k"] = prm.get_double("K");
        bv_params["eps"] = prm.get_double("Eps");
        prm.leave_subsection();

        prm.enter_subsection("Perturbative two defect");
        bv_params["defect-distance"] = prm.get_double("Defect distance");
        bv_params["defect-position-name"] = prm.get("Defect position name");
        bv_params["defect-isomorph-name"] = prm.get("Defect isomorph name");
        prm.leave_subsection();

        prm.leave_subsection();
        // auto boundary_value_func = BoundaryValuesFactory::
        //     BoundaryValuesFactory<dim>(bv_params);
        auto boundary_value_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(am);

        prm.enter_subsection("Initial values");
        std::map<std::string, boost::any> in_params;

        prm.enter_subsection("Boundary values");
        in_params["boundary-values-name"] = prm.get("Name");
        in_params["boundary-condition"] = prm.get("Boundary condition");
        in_params["S-value"] = prm.get_double("S value");

        prm.enter_subsection("Defect configurations");
        in_params["defect-positions"] 
            = ParameterParser::
              parse_coordinate_list<dim>(prm.get("Defect positions"));
        in_params["defect-charges"]
            = ParameterParser::
              parse_number_list(prm.get("Defect charges"));
        in_params["defect-orientations"]
            = ParameterParser::
              parse_number_list(prm.get("Defect orientations"));
        in_params["defect-radius"] = prm.get_double("Defect radius");
        in_params["defect-charge-name"] = prm.get("Defect charge name");
        prm.leave_subsection();

        prm.enter_subsection("Dzyaloshinskii");
        in_params["anisotropy-eps"] = prm.get_double("Anisotropy eps");
        in_params["degree"] = prm.get_integer("Degree");
        in_params["charge"] = prm.get_double("Charge");
        in_params["n-refines"] = prm.get_integer("N refines");
        in_params["tol"] = prm.get_double("Tol");
        in_params["max-iter"] = prm.get_integer("Max iter");
        in_params["newton-step"] = prm.get_double("Newton step");
        prm.leave_subsection();

        prm.enter_subsection("Periodic configurations");
        in_params["phi"] = prm.get_double("Phi");
        in_params["k"] = prm.get_double("K");
        in_params["eps"] = prm.get_double("Eps");
        prm.leave_subsection();

        prm.enter_subsection("Perturbative two defect");
        in_params["defect-distance"] = prm.get_double("Defect distance");
        in_params["defect-position-name"] = prm.get("Defect position name");
        in_params["defect-isomorph-name"] = prm.get("Defect isomorph name");
        prm.leave_subsection();

        prm.leave_subsection();
        auto initial_value_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(in_params);
        prm.leave_subsection();

        prm.enter_subsection("Internal boundary values");
        prm.enter_subsection("Left");
        std::map<std::string, boost::any> l_in_params;

        prm.enter_subsection("Boundary values");
        l_in_params["boundary-values-name"] = prm.get("Name");
        l_in_params["boundary-condition"] = prm.get("Boundary condition");
        l_in_params["S-value"] = prm.get_double("S value");

        prm.enter_subsection("Defect configurations");
        l_in_params["defect-positions"] 
            = ParameterParser::
              parse_coordinate_list<dim>(prm.get("Defect positions"));
        l_in_params["defect-charges"]
            = ParameterParser::
              parse_number_list(prm.get("Defect charges"));
        l_in_params["defect-orientations"]
            = ParameterParser::
              parse_number_list(prm.get("Defect orientations"));
        l_in_params["defect-radius"] = prm.get_double("Defect radius");
        l_in_params["defect-charge-name"] = prm.get("Defect charge name");
        prm.leave_subsection();

        prm.enter_subsection("Dzyaloshinskii");
        l_in_params["anisotropy-eps"] = prm.get_double("Anisotropy eps");
        l_in_params["degree"] = prm.get_integer("Degree");
        l_in_params["charge"] = prm.get_double("Charge");
        l_in_params["n-refines"] = prm.get_integer("N refines");
        l_in_params["tol"] = prm.get_double("Tol");
        l_in_params["max-iter"] = prm.get_integer("Max iter");
        l_in_params["newton-step"] = prm.get_double("Newton step");
        prm.leave_subsection();

        prm.enter_subsection("Periodic configurations");
        l_in_params["phi"] = prm.get_double("Phi");
        l_in_params["k"] = prm.get_double("K");
        l_in_params["eps"] = prm.get_double("Eps");
        prm.leave_subsection();

        prm.enter_subsection("Perturbative two defect");
        l_in_params["defect-distance"] = prm.get_double("Defect distance");
        l_in_params["defect-position-name"] = prm.get("Defect position name");
        l_in_params["defect-isomorph-name"] = prm.get("Defect isomorph name");
        prm.leave_subsection();

        prm.leave_subsection();
        auto left_internal_boundary_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(l_in_params);
        prm.leave_subsection();
        prm.enter_subsection("Right");
        std::map<std::string, boost::any> r_in_params;

        prm.enter_subsection("Boundary values");
        r_in_params["boundary-values-name"] = prm.get("Name");
        r_in_params["boundary-condition"] = prm.get("Boundary condition");
        r_in_params["S-value"] = prm.get_double("S value");

        prm.enter_subsection("Defect configurations");
        r_in_params["defect-positions"] 
            = ParameterParser::
              parse_coordinate_list<dim>(prm.get("Defect positions"));
        r_in_params["defect-charges"]
            = ParameterParser::
              parse_number_list(prm.get("Defect charges"));
        r_in_params["defect-orientations"]
            = ParameterParser::
              parse_number_list(prm.get("Defect orientations"));
        r_in_params["defect-radius"] = prm.get_double("Defect radius");
        r_in_params["defect-charge-name"] = prm.get("Defect charge name");
        prm.leave_subsection();

        prm.enter_subsection("Dzyaloshinskii");
        r_in_params["anisotropy-eps"] = prm.get_double("Anisotropy eps");
        r_in_params["degree"] = prm.get_integer("Degree");
        r_in_params["charge"] = prm.get_double("Charge");
        r_in_params["n-refines"] = prm.get_integer("N refines");
        r_in_params["tol"] = prm.get_double("Tol");
        r_in_params["max-iter"] = prm.get_integer("Max iter");
        r_in_params["newton-step"] = prm.get_double("Newton step");
        prm.leave_subsection();

        prm.enter_subsection("Periodic configurations");
        r_in_params["phi"] = prm.get_double("Phi");
        r_in_params["k"] = prm.get_double("K");
        r_in_params["eps"] = prm.get_double("Eps");
        prm.leave_subsection();

        prm.enter_subsection("Perturbative two defect");
        r_in_params["defect-distance"] = prm.get_double("Defect distance");
        r_in_params["defect-position-name"] = prm.get("Defect position name");
        r_in_params["defect-isomorph-name"] = prm.get("Defect isomorph name");
        prm.leave_subsection();

        prm.leave_subsection();
        auto right_internal_boundary_func = BoundaryValuesFactory::
            BoundaryValuesFactory<dim>(r_in_params);
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
