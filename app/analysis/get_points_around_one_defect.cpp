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
                              const std::vector<std::string> &timestep_names,
                              const std::vector<std::string> &archive_names,
                              const dealii::FullMatrix<double> &centers,
                              unsigned int refinement_level,
                              bool allow_merge,
                              unsigned int max_boxes,
                              const MPI_Comm &mpi_communicator)
{
    NematicSystemMPIDriver<dim> nematic_driver;
    std::unique_ptr<NematicSystemMPI<dim>> nematic_system;

    unsigned int this_process 
        = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    dealii::ConditionalOStream pcout(std::cout, (this_process == 0));

    for (std::size_t time_idx = 0; 
         time_idx < timestep_names.size(); 
         ++time_idx)
    {
        pcout << "Timestep: " << time_idx << "/" 
              << timestep_names.size() << "\n";

        dealii::HDF5::Group timestep_group 
            = file.open_group(timestep_names[time_idx]);
        dealii::HDF5::DataSet Q_data = timestep_group.open_dataset("Q_vec");

        double r0 = Q_data.get_attribute<double>("r0");
        double rf = Q_data.get_attribute<double>("rf");
        unsigned int m = Q_data.get_attribute<unsigned int>("n_r");
        unsigned int n = Q_data.get_attribute<unsigned int>("n_theta");

        std::vector<double> r = NumericalTools::linspace(r0, rf, m);
        std::vector<double> theta = NumericalTools::linspace(0, 2 * M_PI, n);
        std::vector<dealii::Point<dim>> p;
        if (this_process == 0)
            p.resize(n);

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
                    p[j][0] = r[i] * std::cos(theta[j]) + centers(time_idx, 0);
                    p[j][1] = r[i] * std::sin(theta[j]) + centers(time_idx, 1);
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
            Q_data.write_none<double>();
        else
            Q_data.write_selection(total_local_values, 
                                   total_local_value_indices);
    }
}



int main(int ac, char* av[])
{
try
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);

    po::options_description desc("Reads description of radial points around "
                                 "a defect, and a defect center, then queries "
                                 "an archive file for the Q-component values "
                                 "at those points.");
    desc.add_options()
        ("help", "produce help message")
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

    dealii::HDF5::File file(vm["h5_filename"].as<std::string>(),
                            dealii::HDF5::File::FileAccessMode::open,
                            mpi_communicator);

    dealii::HDF5::DataSet centers_dataset = file.open_dataset("centers");
    dealii::HDF5::DataSet times_dataset = file.open_dataset("times");
    dealii::FullMatrix<double> centers
        = centers_dataset.read<dealii::FullMatrix<double>>();
    std::vector<unsigned int> times
        = times_dataset.read<std::vector<unsigned int>>();

    std::vector<std::string> timestep_names(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
        timestep_names[i] = "timestep_" + std::to_string(times[i]);
    std::vector<std::string> archive_names(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
        archive_names[i] = vm["archive_prefix"].as<std::string>() 
                           + std::to_string(times[i]);

    dealii::HDF5::Group timestep_group 
        = file.open_group(timestep_names[0]);
    dealii::HDF5::DataSet Q_data = timestep_group.open_dataset("Q_vec");
    int dim = Q_data.get_attribute<int>("dim");

    if (dim == 2)
    {
        const int dim = 2;

        get_radial_defect_points<dim>(file,
                                      timestep_names,
                                      archive_names,
                                      centers,
                                      vm["refinement_level"]
                                      .as<unsigned int>(),
                                      vm["allow_merge"].as<bool>(),
                                      vm["max_boxes"].as<unsigned int>(),
                                      mpi_communicator);
    }
    else if (dim == 3)
    {
        const int dim = 3;

        get_radial_defect_points<dim>(file,
                                      timestep_names,
                                      archive_names,
                                      centers,
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
