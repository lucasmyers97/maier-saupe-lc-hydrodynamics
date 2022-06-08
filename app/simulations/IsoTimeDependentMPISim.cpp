#include <deal.II/base/utilities.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

#include "LiquidCrystalSystems/IsoTimeDependentMPI.hpp"
#include "Utilities/SimulationOptions.hpp"

namespace po = boost::program_options;

int main(int ac, char* av[])
{
    // TODO: Figure out how to set dim and order at runtime
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message");

    po::variables_map vm = SimulationOptions::read_command_line_options(ac, av);

    if (vm.count("help"))
    {
        return 0;
    }

  const int order = 974;

  try
  {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

      if (vm["simulation-dim"].as<int>() == 2)
      {
          IsoTimeDependentMPI<2, order> iso_time_dependent_mpi(vm);
          iso_time_dependent_mpi.run();
      }
      else if (vm["simulation-dim"].as<int>() == 3)
      {
          IsoTimeDependentMPI<3, order> iso_time_dependent_mpi(vm);
          iso_time_dependent_mpi.run();
      }
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
