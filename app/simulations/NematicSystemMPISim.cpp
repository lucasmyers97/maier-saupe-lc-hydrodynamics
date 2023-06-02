#include "Utilities/ParameterParser.hpp"
#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/base/parameter_handler.h>

#include <string>

int main(int ac, char* av[])
{
    try
    {
        if (ac - 1 != 1)
            throw std::invalid_argument("Error! Didn't input filename");
        std::string parameter_filename(av[1]);

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

        const int dim = 2;

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

        auto nematic_system = std::make_unique<NematicSystemMPI<dim>>(degree);
        nematic_system->get_parameters(prm);

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
        nematic_driver.run(prm);

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
