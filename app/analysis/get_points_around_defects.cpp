#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/base/utilities.h>

#include <mpi.h>
#include <stdexcept>
#include <string>
#include <cmath>

#include "Numerics/NumericalTools.hpp"

namespace po = boost::program_options;

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
        ("center", po::value<std::vector<double>>()->multitoken(),
         "coordinates of defect center point")
        ("n_r", po::value<unsigned int>(), "number of radial points")
        ("n_theta", po::value<unsigned int>(), "number of azimuthal points")
        ("archive_filename", po::value<std::string>(), "name of archive file")
        ("h5_filename", po::value<std::string>(), 
         "name of h5 file, must exist prior to calling function")
        ("h5_groupname", po::value<std::string>(), 
         "name of group in h5 file, must exist prior to calling function")
        ("h5_datasetname", po::value<std::string>(), 
         "name of dataset, must not exist prior to calling function")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);

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

        std::vector<double> center = vm["center"].as<std::vector<double>>();
        if (center.size() != vm["dim"].as<int>())
            throw std::invalid_argument("defect center has wrong number of "
                                        "coordinates");

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        dealii::HDF5::File file(vm["h5_filename"].as<std::string>(),
                                dealii::HDF5::File::FileAccessMode::open,
                                mpi_communicator);
        dealii::HDF5::Group group 
            = file.open_group(vm["h5_groupname"].as<std::string>());

        if (vm["dim"].as<int>() == 2)
        {
            const int dim = 2;

            dealii::Point<dim> defect_pt(center[0], center[1]);
            std::vector<dealii::Point<dim>> p(r.size() * theta.size());
            for (std::size_t i = 0; i < theta.size(); ++i)
                for (std::size_t j = 0; j < r.size(); ++j)
                {
                    p[i * r.size() + j][0] = r[j] * std::cos(theta[i]);
                    p[i * r.size() + j][1] = r[j] * std::sin(theta[i]);
                    p[i * r.size() + j] += defect_pt;
                }

            std::vector<hsize_t> dataset_dimensions = {p.size(), 
                                                       msc::vec_dim<dim>};
            dealii::HDF5::DataSet function_values 
                = group.create_dataset<double>(vm["h5_datasetname"].as<std::string>(), 
                                               dataset_dimensions);

            NematicSystemMPIDriver<dim> nematic_driver;
            nematic_driver
                .read_configuration_at_points(vm["archive_filename"].as<std::string>(), 
                                              p,
                                              function_values);
        }
        else if (vm["dim"].as<int>() == 3)
        {
            const int dim = 3;

            dealii::Point<dim> defect_pt(center[0], center[1], center[2]);
            std::vector<dealii::Point<dim>> p(r.size() * theta.size());
            for (std::size_t i = 0; i < theta.size(); ++i)
                for (std::size_t j = 0; j < r.size(); ++j)
                {
                    p[i * r.size() + j][0] = r[j] * std::cos(theta[i]);
                    p[i * r.size() + j][1] = r[j] * std::sin(theta[i]);
                    p[i * r.size() + j] += defect_pt;
                }

            std::vector<hsize_t> dataset_dimensions = {p.size(), 
                                                       msc::vec_dim<dim>};
            dealii::HDF5::DataSet function_values 
                = group.create_dataset<double>("Q_vec", 
                                               dataset_dimensions);

            NematicSystemMPIDriver<dim> nematic_driver;
            nematic_driver
                .read_configuration_at_points(vm["archive_filename"].as<std::string>(), 
                                              p,
                                              function_values);
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
