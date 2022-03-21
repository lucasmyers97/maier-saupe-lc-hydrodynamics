#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <iostream>
#include <fstream>
#include <cmath>

#include "LiquidCrystalSystems/IsoTimeDependentHydro.hpp"
#include "Utilities/SimulationOptions.hpp"

namespace po = boost::program_options;

int main(int ac, char* av[])
{
    const int dim = 2;
    const int order = 974;
    try
    {
        po::variables_map vm = SimulationOptions::read_command_line_options(ac, av);

        IsoTimeDependentHydro<dim, order> iso_time_dependent_hydro(vm);
        iso_time_dependent_hydro.run();
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
