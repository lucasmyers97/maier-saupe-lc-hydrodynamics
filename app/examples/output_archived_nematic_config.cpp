/**
 * This program reads in a serialization of a NematicSystemMPI from the
 * first command-line argument, and then outputs the configuration into a 
 * vtu file whose filename is based on the second argument.
 */

#include "Numerics/FindLocalMinima.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>

#include <string>

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Utilities/Serialization.hpp"

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Utilities/Serialization.hpp"

int main(int ac, char* av[])
{
    try
    {
        if (ac != 3)
            throw std::invalid_argument("Error! Need to enter input "
                                        "and output filename");

        std::string input_filename(av[1]);
        std::string output_filename(av[2]);

        const int dim = 2;

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        unsigned int degree;
        dealii::Triangulation<dim> coarse_tria;
        dealii::parallel::distributed::Triangulation<dim> tria(mpi_communicator);

        std::unique_ptr<NematicSystemMPI<dim>> nematic_system
            = Serialization::deserialize_nematic_system(mpi_communicator,
                                                        input_filename,
                                                        degree,
                                                        coarse_tria,
                                                        tria);

        const dealii::TrilinosWrappers::MPI::Vector &solution
            = nematic_system->return_current_solution();
        const dealii::DoFHandler<dim> &dof_handler
            = nematic_system->return_dof_handler();

        nematic_system->output_results(mpi_communicator, 
                                       tria, 
                                       "./", 
                                       output_filename, 
                                       0);

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
