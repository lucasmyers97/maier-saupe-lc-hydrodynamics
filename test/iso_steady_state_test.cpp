#include <boost/test/tools/interface.hpp>
#include <boost/test/unit_test_suite.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/program_options.hpp>

#include <fstream>

#define private public
#include "LiquidCrystalSystems/IsoSteadyState.hpp"

namespace utf = boost::unit_test;
namespace po = boost::program_options;

BOOST_AUTO_TEST_CASE(iso_steady_state_archive_test, *utf::tolerance(1e-9))
{
    std::string filename = "iso_steady_state_test.dat";

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
    ("num-refines", po::value<int>()->default_value(2),
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
  ;

  int ac = 1;
  const char *av[2] = {"program", NULL};

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

	const int dim = 2;
  const int order = 590;
  IsoSteadyState<dim, order> iso_steady_state(vm);
  iso_steady_state.run();

  std::ofstream ofs(filename);
  {
      boost::archive::text_oarchive oa(ofs);
      oa << iso_steady_state;
  }

  po::options_description desc2("Allowed options");
  desc2.add_options()
      ("help", "produce help message")

      // Set BoundaryValues parameters
      ("boundary-values-name",
       po::value<std::string>()->default_value("uniform"),
       "sets boundary value scheme")
      ("S-value", po::value<double>()->default_value(0.6751),
       "sets S value at the boundaries")
      ("phi-value", po::value<double>()->default_value(0.732),
       "sets phi value for uniform configurations")
      ("defect-charge-name",
       po::value<std::string>()->default_value("plus-half"),
       "sets defect charge of initial configuration")

      // Set LagrangeMultiplier parameters
      ("lagrange-step-size", po::value<double>()->default_value(0.2),
       "step size of Newton's method for Lagrange Multiplier scheme")
      ("lagrange-max-iters", po::value<int>()->default_value(10),
       "maximum iterations for Newton's method in Lagrange Multiplier scheme")
      ("lagrange-tol", po::value<double>()->default_value(1e-5),
       "tolerance of squared norm in Lagrange Multiplier scheme")

      // Set domain parameters
      ("left-endpoint", po::value<double>()->default_value(-10),
       "left endpoint of square domain grid")
      ("right-endpoint", po::value<double>()->default_value(10),
       "right endpoint of square domain grid")
      ("num-refines", po::value<int>()->default_value(2),
       "number of times to refine domain grid")

      // Set simulation Newton's method parameters
      ("simulation-step-size", po::value<double>()->default_value(2.0),
       "step size for simulation-level Newton's method")
      ("simulation-tol", po::value<double>()->default_value(1e-3),
       "tolerance of normed residual for simulation-level Newton's method")
      ("simulation-max-iters", po::value<int>()->default_value(20),
       "maximum iterations for simulation-level Newton's method")
      ("maier-saupe-alpha", po::value<double>()->default_value(4.0),
       "alpha constant in Maier-Saupe free energy")

      // Set data output parameters
      ("data-folder", po::value<std::string>()->default_value("./blah"),
       "path to folder where output data will be saved")
      ("initial-config-filename",
       po::value<std::string>()->default_value("initial.vtu"),
       "filename of initial configuration data")
      ("final-config-filename",
       po::value<std::string>()->default_value("final.vtu"),
       "filename of final configuration data")
      ("archive-filename",
       po::value<std::string>()->default_value("iso-steady.dat"),
       "filename of archive of IsoSteadyState class");

  po::variables_map vm2;
  po::store(po::parse_command_line(ac, av, desc2), vm2);
  po::notify(vm2);
  
  IsoSteadyState<dim, order> iso_steady_state2(vm2);
  // IsoSteadyState<dim, order> iso_steady_state2;

  {
      std::ifstream ifs(filename);
      boost::archive::text_iarchive ia(ifs);
      ia >> iso_steady_state2;
  }

  BOOST_TEST(iso_steady_state2.left_endpoint == iso_steady_state.left_endpoint);
  BOOST_TEST((iso_steady_state2.fe == iso_steady_state.fe));
}
