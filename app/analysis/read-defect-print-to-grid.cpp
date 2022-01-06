#include <boost/archive/binary_iarchive.hpp>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataSpace.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "maier_saupe_constants.hpp"

#define private public
#include "LiquidCrystalSystems/IsoSteadyState.hpp"

namespace msc = maier_saupe_constants;








// template <int dim>
// void  EvaluateFEObject<dim>::run()
// {
//     constexpr int order = 590;
//     std::string ms_folder = "/home/lucas/Documents/grad-work/research/"
//                             "maier-saupe-lc-hydrodynamics/";

//     std::string grid_filename = ms_folder + "data/simulations/"
//                                 "iso-steady-state/cody-data/"
//                                 "plus-half-defect-cody.h5"; 
//     read_grid(grid_filename);

//     // std::string simulation_filename = ms_folder + "data/simulations/"
//     //                                               "iso-steady-state/2021-12-07/"
//     //                                               "plus-half-iso-steady-state.dat";
//     std::string simulation_filename = ms_folder + "iso-steady-state.dat";
//     std::cout << simulation_filename << "\n";
//     IsoSteadyState<dim, order> iso_steady_state;

//     std::ifstream ifs(simulation_filename);
//     boost::archive::text_iarchive ia(ifs);
//     iso_steady_state.load(ia, 0);

//     // read_fe_at_points(iso_steady_state.dof_handler,
//     //                   iso_steady_state.current_solution);

//     // std::string output_filename = ms_folder + "data/simulations/"
//     //                                           "iso-steady-state/2021-12-07/"
//     //                                           "plus-half-defect.h5";
//     // write_values_to_grid(output_filename);
// }



int main()
{
    const int dim = 2;
    std::vector<std::string> meshgrid_names(dim);
    meshgrid_names[0] = "X";
    meshgrid_names[1] = "Y";

    EvaluateFEObject<dim> defect_to_grid(meshgrid_names);
    defect_to_grid.run();

    return 0;
}
