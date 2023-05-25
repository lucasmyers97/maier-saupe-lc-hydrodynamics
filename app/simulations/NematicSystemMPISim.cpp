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

        const int dim = 3;

        dealii::ParameterHandler prm;
        std::ifstream ifs(parameter_filename);
        NematicSystemMPIDriver<dim>::declare_parameters(prm);
        NematicSystemMPI<dim>::declare_parameters(prm);
        prm.parse_input(ifs);

        NematicSystemMPIDriver<dim> nematic_driver;
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
