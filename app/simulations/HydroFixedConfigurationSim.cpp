#include "SimulationDrivers/BasicHydroDriver.hpp"

#include <deal.II/base/function.h>

#include <memory>

#include "ExampleFunctions/PlusHalfActiveSource.hpp"

int main(int ac, char* av[])
{
    try
    {
        const int dim = 2;

        std::unique_ptr<dealii::Function<dim>> forcing_function =
            std::make_unique<PlusHalfActiveSource<dim>>();

        BasicHydroDriver<dim> hydro_driver(std::move(forcing_function));
        hydro_driver.run();
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
