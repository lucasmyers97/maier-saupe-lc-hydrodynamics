#include "LiquidCrystalSystems/DzyaloshinskiiSystem.hpp"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <string>
#include <ostream>

int main(int ac, char* av[])
{
    /* Read command line options */
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("epsilon", 
         po::value<double>()->default_value(0.0), 
         "Elastic anisotropy parameter")
        ("n_points", 
         po::value<unsigned int>()->default_value(100),
         "Number of points to produce for hdf5 file")

        ("n_refines",
         po::value<unsigned int>()->default_value(8),
         "Number of times to subdivide domain")
        ("degree",
         po::value<unsigned int>()->default_value(1),
         "Finite element degree")
        ("tolerance",
         po::value<double>()->default_value(1e-8),
         "Tolerance for L2 norm of residual")
        ("max_iters",
         po::value<unsigned int>()->default_value(100),
         "Maximum number of Newton iterations")
        ("newton_step",
         po::value<double>()->default_value(1.0),
         "Step size for each Newton iteration")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }


    double eps = vm["epsilon"].as<double>();
    unsigned int n_points = vm["n_points"].as<unsigned int>();

    unsigned int degree = vm["degree"].as<unsigned int>();
    unsigned int n_refines = vm["n_refines"].as<unsigned int>();
    double tol = vm["tolerance"].as<double>();
    unsigned int max_iter = vm["max_iters"].as<unsigned int>();
    double newton_step = vm["newton_step"].as<double>();

    std::string filename("dzyaloshinskii_solution_");
    filename += std::to_string(eps);

    std::string vtu_filename = filename + std::string(".vtu");
    std::string hdf5_filename = filename + std::string(".h5");

    DzyaloshinskiiSystem dzyaloshinskii_system(eps, degree);
    dzyaloshinskii_system.make_grid(n_refines);
    dzyaloshinskii_system.setup_system();
    dzyaloshinskii_system.run_newton_method(tol, max_iter, newton_step);
    dzyaloshinskii_system.output_solution(vtu_filename);
    dzyaloshinskii_system.output_hdf5(n_points, hdf5_filename);

    return 0;
}
