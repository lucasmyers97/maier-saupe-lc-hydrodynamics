#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/base/parameter_handler.h>

#include <string>

int main(int ac, char* av[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

    const int dim = 2;
    std::string parameter_filename("nematic_mpi_parameters.prm");

    NematicSystemMPIDriver<dim> nematic_driver;
    nematic_driver.run(parameter_filename);

    return 0;
}
