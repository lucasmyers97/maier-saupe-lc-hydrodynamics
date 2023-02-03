#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/hdf5.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/full_matrix.h>

#include <mpi.h>
#include <stdexcept>
#include <string>
#include <cmath>

#include "Numerics/NumericalTools.hpp"

namespace po = boost::program_options;

template <int dim>
void get_radial_defect_points(dealii::HDF5::File &file,
                              const std::vector<double> &r, 
                              const std::vector<double> &theta,
                              const std::vector<std::string> &timestep_names,
                              const std::vector<std::string> &archive_names,
                              const dealii::FullMatrix<double> &pos_centers,
                              const dealii::FullMatrix<double> &neg_centers,
                              unsigned int refinement_level,
                              bool allow_merge,
                              unsigned int max_boxes,
                              const MPI_Comm &mpi_communicator)
{
    NematicSystemMPIDriver<dim> nematic_driver;
    std::unique_ptr<NematicSystemMPI<dim>> nematic_system;

    std::size_t m = r.size();
    std::size_t n = theta.size();

    std::vector<hsize_t> dataset_dimensions = {m * n, msc::vec_dim<dim>};
    std::vector<dealii::Point<dim>> p;
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        p.resize(n);

    dealii::ConditionalOStream 
        pcout(std::cout,
              (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

    for (std::size_t time_idx = 0; 
         time_idx < timestep_names.size(); 
         ++time_idx)
    {
        pcout << "Timestep: " << time_idx << "/" 
              << timestep_names.size() << "\n";

        dealii::HDF5::Group timestep_group 
            = file.open_group(timestep_names[time_idx]);
        dealii::HDF5::DataSet pos_data
            = timestep_group.create_dataset<double>("pos_Q_vec", 
                                                    dataset_dimensions);
        dealii::HDF5::DataSet neg_data
            = timestep_group.create_dataset<double>("neg_Q_vec", 
                                                    dataset_dimensions);

        nematic_system = nematic_driver.deserialize(archive_names[time_idx]);

        // setup grid for getting points
        auto cache = nematic_driver.get_grid_cache();
        auto bounding_boxes 
            = nematic_driver.get_bounding_boxes(refinement_level,
                                                allow_merge,
                                                max_boxes);
        auto global_bounding_boxes 
            = dealii::GridTools::
              exchange_local_bounding_boxes(bounding_boxes, mpi_communicator);

        std::vector<double> local_values;
        std::vector<hsize_t> local_value_indices;
        std::vector<double> total_local_values;
        std::vector<hsize_t> total_local_value_indices;
        hsize_t offset = 0;
        for (std::size_t i = 0; i < m; ++i)
        {
            offset = i * n;

            // get points for this timestep, and this r-value
            if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                for (std::size_t j = 0; j < n; ++j)
                {
                    p[j][0] = r[i] * std::cos(theta[j]) + pos_centers(time_idx, 0);
                    p[j][1] = r[i] * std::sin(theta[j]) + pos_centers(time_idx, 1);
                }

            std::tie(local_values, local_value_indices)
                = nematic_driver.
                  read_configuration_at_points(*nematic_system,
                                               p, 
                                               cache, 
                                               global_bounding_boxes,
                                               offset);

            // concatenate local values corresponding to const r slice
            // to the vector holding *all* locally-held points
            total_local_values.insert(total_local_values.end(),
                                      local_values.begin(),
                                      local_values.end());
            total_local_value_indices.insert(total_local_value_indices.end(),
                                             local_value_indices.begin(),
                                             local_value_indices.end());
        }
        if (total_local_values.empty())
            pos_data.write_none<double>();
        else
            pos_data.write_selection(total_local_values, 
                                     total_local_value_indices);
        total_local_values.clear();
        total_local_value_indices.clear();
        for (std::size_t i = 0; i < m; ++i)
        {
            offset = i * n;

            // get points for this timestep, and this r-value
            if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                for (std::size_t j = 0; j < n; ++j)
                {
                    p[j][0] = r[i] * std::cos(theta[j]) + neg_centers(time_idx, 0);
                    p[j][1] = r[i] * std::sin(theta[j]) + neg_centers(time_idx, 1);
                }

            std::tie(local_values, local_value_indices)
                = nematic_driver.
                  read_configuration_at_points(*nematic_system,
                                               p, 
                                               cache, 
                                               global_bounding_boxes,
                                               offset);

            // concatenate local values corresponding to const r slice
            // to the vector holding *all* locally-held points
            total_local_values.insert(total_local_values.end(),
                                      local_values.begin(),
                                      local_values.end());
            total_local_value_indices.insert(total_local_value_indices.end(),
                                             local_value_indices.begin(),
                                             local_value_indices.end());
        }
        if (total_local_values.empty())
            neg_data.write_none<double>();
        else
            neg_data.write_selection(total_local_values, 
                                     total_local_value_indices);
    }
}



int main(int ac, char* av[])
{
    po::options_description desc("Reads description of radial points around "
                                 "a defect, and a defect center, then queries "
                                 "an archive file for the Q-component values "
                                 "at those points.");
    desc.add_options()
        ("help", "produce help message")
        ("dim", po::value<int>(), "dimension of simulation")
        ("r0", po::value<double>(), "inner radius at which points are sampled")
        ("rf", po::value<double>(), "outer radius at which points are sampled")
        ("n_r", po::value<unsigned int>(), "number of radial points")
        ("n_theta", po::value<unsigned int>(), "number of azimuthal points")
        ("archive_prefix", po::value<std::string>(), "prefix for archive file")
        ("h5_filename", po::value<std::string>(), 
         "name of h5 file, must exist prior to calling function")
        ("refinement_level", po::value<unsigned int>()->default_value(3),
         "Refinement level for boxes bounding locally-owned cells")
        ("allow_merge", po::value<bool>()->default_value(false), 
         "whether to allow merging of bouding boxes")
        ("max_boxes", po::value<unsigned int>()
                      ->default_value(dealii::numbers::invalid_unsigned_int),
         "maximum number of bounding boxes for a locally-owned domain")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc, 
                                     po::command_line_style::unix_style 
                                     ^ po::command_line_style::allow_short), 
              vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }


    try
    {
        std::vector<double> r 
            = NumericalTools::linspace(vm["r0"].as<double>(),
                                       vm["rf"].as<double>(),
                                       vm["n_r"].as<unsigned int>());
        std::vector<double> theta
            = NumericalTools::linspace(0, 
                                       2 * M_PI,
                                       vm["n_theta"].as<unsigned int>());

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        dealii::HDF5::File file(vm["h5_filename"].as<std::string>(),
                                dealii::HDF5::File::FileAccessMode::open,
                                mpi_communicator);
        dealii::HDF5::DataSet pos_centers_dataset
            = file.open_dataset("pos_centers");
        dealii::HDF5::DataSet neg_centers_dataset
            = file.open_dataset("neg_centers");
        dealii::HDF5::DataSet times_dataset
            = file.open_dataset("times");

        dealii::FullMatrix<double> pos_centers
            = pos_centers_dataset.read<dealii::FullMatrix<double>>();
        dealii::FullMatrix<double> neg_centers
            = neg_centers_dataset.read<dealii::FullMatrix<double>>();
        std::vector<unsigned int> times
            = times_dataset.read<std::vector<unsigned int>>();

        std::vector<std::string> timestep_names(times.size());
        for (std::size_t i = 0; i < times.size(); ++i)
            timestep_names[i] = "timestep_" + std::to_string(times[i]);
        std::vector<std::string> archive_names(times.size());
        for (std::size_t i = 0; i < times.size(); ++i)
            archive_names[i] = vm["archive_prefix"].as<std::string>() 
                               + std::to_string(times[i]);

        if (vm["dim"].as<int>() == 2)
        {
            const int dim = 2;

            get_radial_defect_points<dim>(file,
                                          r, 
                                          theta,
                                          timestep_names,
                                          archive_names,
                                          pos_centers,
                                          neg_centers,
                                          vm["refinement_level"]
                                          .as<unsigned int>(),
                                          vm["allow_merge"].as<bool>(),
                                          vm["max_boxes"].as<unsigned int>(),
                                          mpi_communicator);
        }
        else if (vm["dim"].as<int>() == 3)
        {
            const int dim = 3;

            get_radial_defect_points<dim>(file,
                                          r, 
                                          theta,
                                          timestep_names,
                                          archive_names,
                                          pos_centers,
                                          neg_centers,
                                          vm["refinement_level"]
                                          .as<unsigned int>(),
                                          vm["allow_merge"].as<bool>(),
                                          vm["max_boxes"].as<unsigned int>(),
                                          mpi_communicator);
        }
        else
            throw std::invalid_argument("Dimension can be 2 or 3");


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
