#include "SimulationDrivers/BasicLiquidCrystalDriver.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>

#include <memory>
#include <vector>

int main(int ac, char* av[])
{
    try
    {
        const int dim = 2;
        const unsigned int num_refines = 9;
        const double left = -233.0;
        const double right = 233.0;
        const double dt = 0.5;
        const double simulation_tol = 1e-8;
        const unsigned int simulation_max_iters = 20;
        const unsigned int n_steps = 20;
        const std::string data_folder = "./";
        const std::string config_filename = "basic_lc_sim_config";

        BasicLiquidCrystalDriver<dim> lc_driver(num_refines,
                                                left,
                                                right,
                                                dt,
                                                simulation_tol,
                                                simulation_max_iters,
                                                n_steps,
                                                data_folder,
                                                config_filename);
        lc_driver.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
