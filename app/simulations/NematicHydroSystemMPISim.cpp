#include "SimulationDrivers/NematicHydroSystemMPIDriver.hpp"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/mpi.h>

#include <exception>
#include <iostream>
#include <istream>
#include <stdexcept>
#include <string>

int main(int ac, char* av[])
{
    try
    {
        const int dim = 2;
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

        if (ac - 1 != 1)
            throw std::invalid_argument("Error! Must input parameter filename");

        std::string parameter_filename(av[1]);

        dealii::ParameterHandler prm;
        NematicHydroSystemMPIDriver<dim>::declare_parameters(prm);

        std::ifstream ifs(parameter_filename);
        prm.parse_input(ifs);

        NematicHydroSystemMPIDriver<dim> nematic_hydro_driver;
        nematic_hydro_driver.run(prm);

        return 0;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

}
