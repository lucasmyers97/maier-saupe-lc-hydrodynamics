#include <deal.II/base/utilities.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

#include "LiquidCrystalSystems/IsoSteadyStateMPI.hpp"

namespace po = boost::program_options;

int main(int ac, char* av[])
{
  // TODO: Figure out how to set dim and order at runtime
    po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")

    // Set BoundaryValues parameters
    ("boundary-values-name",
     po::value<std::string>()->default_value("defect"),
     "sets boundary value scheme")
    ("S-value", po::value<double>()->default_value(0.6751),
     "sets S value at the boundaries")
    ("defect-charge-name",
     po::value<std::string>()->default_value("plus-half"),
     "sets defect charge of initial configuration")

    // Set LagrangeMultiplier parameters
    ("lagrange-step-size", po::value<double>()->default_value(1.0),
     "step size of Newton's method for Lagrange Multiplier scheme")
    ("lagrange-max-iters", po::value<int>()->default_value(20),
     "maximum iterations for Newton's method in Lagrange Multiplier scheme")
    ("lagrange-tol", po::value<double>()->default_value(1e-8),
     "tolerance of squared norm in Lagrange Multiplier scheme")

    // Set domain parameters
    ("left-endpoint", po::value<double>()->default_value(-10 / std::sqrt(2)),
     "left endpoint of square domain grid")
    ("right-endpoint", po::value<double>()->default_value(10 / std::sqrt(2)),
     "right endpoint of square domain grid")
    ("num-refines", po::value<int>()->default_value(4),
     "number of times to refine domain grid")

    // Set simulation Newton's method parameters
    ("simulation-step-size", po::value<double>()->default_value(1.0),
     "step size for simulation-level Newton's method")
    ("simulation-tol", po::value<double>()->default_value(1e-8),
     "tolerance of normed residual for simulation-level Newton's method")
    ("simulation-max-iters", po::value<int>()->default_value(10),
     "maximum iterations for simulation-level Newton's method")
    ("maier-saupe-alpha", po::value<double>()->default_value(8.0),
     "alpha constant in Maier-Saupe free energy")

    // Set data output parameters
    ("data-folder",
     po::value<std::string>()->default_value("./"),
     "path to folder where output data will be saved")
    ("initial-config-filename",
     po::value<std::string>()->default_value("initial-configuration.vtu"),
     "filename of initial configuration data")
    ("final-config-filename",
     po::value<std::string>()->default_value("final-configuration.vtu"),
     "filename of final configuration data")
    ("archive-filename",
     po::value<std::string>()->default_value("iso-steady-state.dat"),
     "filename of archive of IsoSteadyState class")
    ("grid-input-filename",
     po::value<std::string>()->default_value("/home/lucas/Documents/grad-work/research/maier-saupe-lc-hydrodynamics/data/simulations/iso-steady-state/cody-data/plus-half-defect-cody.h5"),
     "filename of hdf5 file holding grid to write to")
    ("grid-output-filename",
     po::value<std::string>()->default_value("iso_steady_state_grid.h5"),
     "filenmae of hdf5 file holding FE object written to grid")
    ("meshgrid-X-name",
     po::value<std::string>()->default_value("X"),
     "name in hdf5 file of grid holding x-coordinates")
    ("meshgrid-Y-name",
     po::value<std::string>()->default_value("Y"),
     "name in hdf5 file of grid holding y-coordinates")
    ("dist-scale",
     po::value<double>()->default_value(1.0 / std::sqrt(2)),
     "coefficient specifying how to scale the input-grid for this system")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
      std::cout << desc << "\n";
      return 0;
  }

	const int dim = 2;
  const int order = 974;

  try
  {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
      IsoSteadyStateMPI<dim, order> iso_steady_state_mpi(vm);
      iso_steady_state_mpi.run();
  }
  catch (std::exception &exc)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  catch (...)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }

        return 0;
}