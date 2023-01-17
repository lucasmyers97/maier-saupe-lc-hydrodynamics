/**
 * Tests compute_set_union function by seeing whether each process prints
 * out the union of the vectors on each of the processes.
 */
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/point.h>

#include <vector>
#include <iostream>

int main(int ac, char* av[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        std::size_t n_entries_per_process = 1;
        constexpr int dim = 2;
        std::vector<std::vector<double>> x(n_entries_per_process,
                                           std::vector<double>(dim));
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            x[0][1] = 2.0;
        else
            x[0][1] = 3.0;

        x = dealii::Utilities::MPI::compute_set_union(x, mpi_communicator);
        
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            for (const auto &entry : x)
            {
                for (const auto &e : entry)
                    std::cout << e << " ";
                std::cout << "\n";
            }
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
