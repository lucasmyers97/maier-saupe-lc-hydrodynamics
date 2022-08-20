#include "Utilities/Output.hpp"

#include <deal.II/base/mpi.h>

#include <vector>
#include <string>
#include <iostream>
#include <random>

int main(int ac, char* av[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        std::vector<std::string> data_names = {"data"};
        std::string filename = "./data_test.h5";

        std::size_t n = {0};
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            n = 100;

        std::minstd_rand rand_func;
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::vector<double>> data_set = {std::vector<double>(n)};
        for (auto &datum : data_set[0])
            datum = dist(rand_func);

        Output::distributed_vector_to_hdf5(data_set, 
                                           data_names, 
                                           mpi_communicator, 
                                           filename);

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;

        return -1;
    }
}
