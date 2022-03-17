#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>

namespace SimulationOptions
{
    namespace po = boost::program_options;

    po::variables_map read_command_line_options(int ac, char *av[])
    {
        po::options_description desc("Allowed options");
        desc.add_options()("help", "produce help message")

        // Set BoundaryValues parameters
        ("boundary-values-name",
         po::value<std::string>()->default_value("defect"),
         "sets boundary value scheme")
        ("S-value", po::value<double>()->default_value(0.6751),
         "sets S value at the boundaries")
        ("defect-charge-name",
         po::value<std::string>()->default_value("plus-half"),
         "sets defect charge of initial configuration")
        ("centers",
         po::value<std::vector<double>>()
         ->default_value(std::vector<double>({-5, 0, 5, 0}), "-5, 0, 5, 0")
         ->multitoken(),
         "sets centers of defects for a two-charge configuration")

        // Set LagrangeMultiplier parameters
        ("lagrange-step-size", po::value<double>()->default_value(1.0),
         "step size of Newton's method for Lagrange Multiplier scheme")
        ("lagrange-max-iters", po::value<int>()->default_value(20),
         "maximum iterations for Newton's method in Lagrange Multiplier "
         "scheme")
        ("lagrange-tol", po::value<double>()->default_value(1e-8),
         "tolerance of squared norm in Lagrange Multiplier scheme")

        // Set domain parameters
        ("left-endpoint",
         po::value<double>()->default_value(-10 / std::sqrt(2)),
         "left endpoint of square domain grid")
        ("right-endpoint",
         po::value<double>()->default_value(10 / std::sqrt(2)),
         "right endpoint of square domain grid")
        ("num-refines", po::value<int>()->default_value(4),
         "number of times to refine domain grid")

        // Set simulation Newton's method parameters
        ("simulation-step-size", po::value<double>()->default_value(1.0),
         "step size for simulation-level Newton's method")
        ("simulation-tol", po::value<double>()->default_value(1e-8),
         "tolerance of normed residual for simulation-level Newton's "
         "method")
        ("simulation-max-iters", po::value<int>()->default_value(10),
         "maximum iterations for simulation-level Newton's method")
        ("maier-saupe-alpha", po::value<double>()->default_value(8.0),
         "alpha constant in Maier-Saupe free energy")
        ("dt", po::value<double>()->default_value(0.1),
         "size of time step")
        ("n-steps", po::value<int>()->default_value(10),
         "number of time steps for simulation")

        // Set data output parameters
        ("data-folder", po::value<std::string>()->default_value("./"),
         "path to folder where output data will be saved")
        ("initial-config-filename",
         po::value<std::string>()->default_value("initial-configuration"),
         "filename of initial configuration data")
        ("final-config-filename",
         po::value<std::string>()->default_value("final-configuration"),
         "filename of final configuration data")
        ("archive-filename",
         po::value<std::string>()->default_value("iso-steady-state.dat"),
         "filename of archive of IsoSteadyState class")
        ("grid-input-filename",
         po::value<std::string>()->default_value(
                "/home/lucas/Documents/grad-work/research/"
                "maier-saupe-lc-hydrodynamics/data/simulations/"
                "iso-steady-state/cody-data/plus-half-defect-cody.h5"),
         "filename of hdf5 file holding grid to write to")
        ("grid-output-filename",
         po::value<std::string>()->default_value("iso_steady_state_grid.h5"),
         "filenmae of hdf5 file holding FE object written to grid")
        ("meshgrid-X-name", po::value<std::string>()->default_value("X"),
         "name in hdf5 file of grid holding x-coordinates")
        ("meshgrid-Y-name", po::value<std::string>()->default_value("Y"),
         "name in hdf5 file of grid holding y-coordinates")
        ("dist-scale", po::value<double>()->default_value(1.0 / std::sqrt(2)),
         "coefficient specifying how to scale the input-grid for this "
         "system");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        return vm;
    }
}
