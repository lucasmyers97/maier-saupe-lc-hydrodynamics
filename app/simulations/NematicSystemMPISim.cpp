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

        NematicSystemMPIDriver<dim> nematic_driver;
        nematic_driver.run(parameter_filename);

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
