#ifndef SIMULATION_OPTIONS_HPP
#define SIMULATION_OPTIONS_HPP

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>

namespace SimulationOptions
{
    namespace po = boost::program_options;

    po::variables_map read_command_line_options(int ac, char *av[]);
}

#endif
