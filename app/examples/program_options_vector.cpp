#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>

#include <vector>
#include <iostream>

int main(int ac, char* av[])
{
    namespace po = boost::program_options;
    po::options_description desc("");

    desc.add_options()(
        "opt",
        po::value<std::vector<double>>()
        ->default_value(std::vector<double>({3.14, 2.76, 1.11111}), "3.14, 2.76, 1.11111")
            ->multitoken(),
        "trying for vector");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    for (auto &vec : vm["opt"].as<std::vector<double>>())
        std::cout << vec << std::endl;
}
