#include "SimulationDrivers/NematicSystemMPIDriver.hpp"
#include "LiquidCrystalSystems/NematicSystemMPI.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <exception>
#include <iostream>
#include <mpi.h>
#include <string>

int main(int ac, char* av[])
{
    const int dim = 2;
    try
    {
        if (ac - 1 != 2)
            throw std::invalid_argument("Error! Didn't input filename");
        std::string filename(av[1]);
        unsigned int n_refinement(std::stoi(av[2]));

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);
        bool is_zero_rank = (dealii::Utilities::MPI::
                             this_mpi_process(mpi_communicator) == 0);
        dealii::ConditionalOStream pcout(std::cout, is_zero_rank);

        NematicSystemMPIDriver<dim> nematic_driver;
        std::unique_ptr<NematicSystemMPI<dim>> nematic_system
            = nematic_driver.deserialize(filename);

        auto cache = nematic_driver.get_grid_cache();
        auto bounding_boxes = nematic_driver.get_bounding_boxes(n_refinement);

        for (const auto &bounding_box : bounding_boxes)
        {
            auto boundary_points = bounding_box.get_boundary_points();
            pcout << boundary_points.first << "\n";
            pcout << boundary_points.second << "\n";
            pcout << "\n";
        }
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Uncaught exception!" << std::endl;
    }

    return 0;
}
