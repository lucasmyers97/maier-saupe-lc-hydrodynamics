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
    po::variables_map vm = SimulationOptions::read_command_line_options(ac, av);

    const int dim = 2;
    const int order = 974;
    IsoTimeDependentHydro<dim, order> iso_time_dependent_hydro(vm);
    iso_time_dependent_hydro.run();

    return 0;
}
