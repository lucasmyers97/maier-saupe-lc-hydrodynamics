/**
 * Program populates each mpi process with arbitrary data, then uses
 * the allgather function to collect the sizes of the data on each process.
 * This allows us to write hyperslabs of the data into an hdf5 file
 * concurrently, since each process knows the offset and dimension of its own
 * data.
 *
 * Usage: mpirun -np 6 ./install/bin/examples/dealii_hdf5_write
 */

#include <deal.II/base/mpi.h>
#include <deal.II/base/hdf5.h>

#include <string>
#include <vector>
#include <numeric>
#include <iterator>

int main(int ac, char* av[])
{

    // start mpi processes, get process number
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    unsigned int mpi_process 
        = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    // generate some arbitrary data of varying lengths across processes
    std::vector<double> data;
    std::vector<double> new_data;
    switch (mpi_process) 
    {
    case 0:
        new_data = {0.0, 1.0, -1.1, 0.5};
        data.insert(data.end(), new_data.begin(), new_data.end());
        break;
    case 1:
        new_data = {-30};
        data.insert(data.end(), new_data.begin(), new_data.end());
        break;
    case 2:
        new_data = {23.0, 21.7};
        data.insert(data.end(), new_data.begin(), new_data.end());
        break;
    default:
        new_data = {0.0, 0.0, 15.1, 0.5};
        data.insert(data.end(), new_data.begin(), new_data.end());
        break;
    }

    // lengths of data from all processes
    std::vector<std::size_t> process_data_length
        = dealii::Utilities::MPI::all_gather(mpi_communicator, data.size());

    // print out lengths and stuff
    if (mpi_process == 0)
        for (const auto num : process_data_length)
            std::cout << num << "\n";

    auto this_process_iter 
        = std::next(process_data_length.begin(), mpi_process);
    unsigned int write_index 
        = std::accumulate(process_data_length.begin(), this_process_iter, 0);

    unsigned int mpi_print_process = 0;
    if (ac > 1)
        mpi_print_process = std::stoi(std::string(av[1]));

    if (mpi_process == mpi_print_process)
        std::cout << "Write index is: " << write_index << "\n";

    // Write to hdf5 file
    hsize_t total_data_length = std::accumulate(process_data_length.begin(), 
                                                process_data_length.end(), 
                                                0);
    std::vector<hsize_t> dataset_dims = {total_data_length};
    std::vector<hsize_t> hyperslab_offset = {write_index};
    std::vector<hsize_t> hyperslab_dims = {process_data_length[mpi_process]};

    std::string filename("hdf5_test.h5");
    std::string group_name("test_group1");
    std::string dataset_name("test_dataset1");

    dealii::HDF5::File file(filename, 
                            dealii::HDF5::File::FileAccessMode::create,
                            mpi_communicator);
    auto group = file.create_group(group_name);
    auto dataset = group.create_dataset<double>(dataset_name, dataset_dims);

    dataset.write_hyperslab(data, hyperslab_offset, hyperslab_dims);

    return 0;
}
