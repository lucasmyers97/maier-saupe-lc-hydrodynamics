#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <boost/program_options.hpp>

#include <string>
#include <vector>
#include <iostream>

namespace po = boost::program_options;
namespace hf = HighFive;

int main(int ac, char *av[])
{
    // Read in filename
    po::options_description desc("Allowed options");
    desc.add_options()
        ("filename", po::value<std::string>(), "hdf5 filename")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    std::string filename(vm["filename"].as<std::string>());

    hf::File f(filename);
    std::vector<std::string> object_names = f.listObjectNames();

    for (const auto &name : object_names)
        std::cout << name << "\n";

    return 0;
}
