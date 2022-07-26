/**
 * Program first populates each mpi process with an arbitrary number, and then
 * generates a vector in each process which is each of those arbitrary numbers
 * concatenated.
 * The purpose of this is to simulate each process having some number of
 * elements it needs to write to an hdf5 file, and gathering that number across
 * each of the processes so that each process knows where to start writing into
 * the HDF5 program.
 *
 * Usage:
 * mpirun -np 6 ./install/bin/examples/dealii_mpi_allgather <process-to-print>
 */

#include <deal.II/base/mpi.h>
#include <deal.II/base/hdf5.h>

#include <string>
#include <vector>
#include <numeric>
#include <iterator>

int main(int ac, char* av[])
{

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);

    unsigned int mpi_process 
        = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    unsigned int arb_number = 0;
    switch (mpi_process) 
    {
    case 0:
        arb_number = 3;
        break;
    case 1:
        arb_number = 2;
        break;
    case 2:
        arb_number = 60;
        break;
    default:
        arb_number = 5;
        break;
    }

    std::vector<unsigned int> arb_vector
        = dealii::Utilities::MPI::all_gather(mpi_communicator, arb_number);

    if (mpi_process == 0)
        for (const auto num : arb_vector)
            std::cout << num << "\n";

    auto this_process_iter = std::next(arb_vector.begin(), mpi_process);
    unsigned int write_index 
        = std::accumulate(arb_vector.begin(), this_process_iter, 0);

    unsigned int mpi_print_process = 0;
    // possible error here due to string not being a number
    if (ac > 1)
        mpi_print_process = std::stoi(std::string(av[1]));

    if (mpi_process == mpi_print_process)
        std::cout << write_index << "\n";

    return 0;
}
