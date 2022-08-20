#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/base/hdf5.h>

#include <stdexcept>
#include <vector>
#include <iostream>
#include <fstream>
#include <exception>

namespace Output {

/**
 * \brief Writes a collection of vectors to hdf5 -- each gets its own dataset
 *
 * @param[in] data_set Each vector in this collection is written to a dataset
 * @param[in] data_names Corresponding names of the datasets
 * @param[in] mpi_communicator 
 * @param[in] filename Name of the hdf5 file which is created and written to
 */
inline void 
distributed_vector_to_hdf5(const std::vector<std::vector<double>> &data_set,
                           const std::vector<std::string> &data_names,
                           const MPI_Comm &mpi_communicator,
                           const std::string &filename)
{
    // each vector corresponds to a dataset
    if (data_set.size() != data_names.size())
    {
        throw std::length_error("Length of data sets" 
                                " and data names don't match");
        return;
    }

    dealii::HDF5::File file(filename, 
                            dealii::HDF5::File::FileAccessMode::create, 
                            mpi_communicator);

    unsigned int this_process 
        = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    // write each data vector into its right place based on mpi process
    for (std::size_t i = 0; i < data_set.size(); ++i)
    {
        std::vector<std::size_t> process_data_lengths
            = dealii::Utilities::MPI::all_gather(mpi_communicator, 
                                                 data_set[i].size());
        auto this_process_iter 
            = std::next(process_data_lengths.begin(), this_process);
        hsize_t write_index 
            = std::accumulate(process_data_lengths.begin(), 
                              this_process_iter, 
                              0);
        hsize_t total_data_length 
            = std::accumulate(process_data_lengths.begin(), 
                              process_data_lengths.end(), 
                              0);

        std::vector<hsize_t> dataset_dims = {total_data_length};
        std::vector<hsize_t> hyperslab_offset = {write_index};
        std::vector<hsize_t> hyperslab_dims 
            = {process_data_lengths[this_process]};

        auto dataset = file.create_dataset<double>(data_names[i], 
                                                   dataset_dims);

        // if there's no data, trying to write to a dataset errors out
        if (total_data_length == 0)
            continue;

        if (process_data_lengths[this_process] == 0)
            dataset.write_none<double>();
        else
            dataset.write_hyperslab(data_set[i], 
                                    hyperslab_offset, 
                                    hyperslab_dims);
    }
}

} // namespace Output

#endif
