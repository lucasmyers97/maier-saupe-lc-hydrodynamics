#ifndef HYDRO_FIXED_LC_DRIVER_HPP
#define HYDRO_FIXED_LC_DRIVER_HPP

#include <deal.II/grid/tria.h>

#include <boost/archive/text_iarchive.hpp>

#include <fstream>
#include <string>
#include <vector>

#include "LiquidCrystalSystems/HydroFixedConfiguration.hpp"
#include "LiquidCrystalSystems/LiquidCrystalSystem.hpp"

template <int dim>
class HydroFixedLCDriver
{
public:
    HydroFixedLCDriver(){};
    void run();

private:
    void deserialize_lc_configuration(std::string filename,
                                      LiquidCrystalSystem<dim> &lc_system);
    void assemble_hydro_system(HydroFixedConfiguration<dim> &hydro_config);

    dealii::Triangulation<dim> tria;
};



template <int dim>
void HydroFixedLCDriver<dim>::
deserialize_lc_configuration(std::string filename,
                             LiquidCrystalSystem<dim> &lc_system)
{
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> tria;
        ia >> lc_system;
    }
}



template <int dim>
void HydroFixedLCDriver<dim>::run()
{
    std::string filename("two_defect_512_512.ar");
    int order = 590;
    unsigned int degree = 2;
    std::string boundary_values_name = "two-defect";
    std::map<std::string, boost::any> am;
    am["S-value"] = 0.6751;
    am["defect-charge-name"] = std::string("plus-half-minus-half");
    am["centers"] = std::vector<double>({-35.0, 0, 35.0, 0});
    double lagrange_step_size = 1.0;
    double lagrange_tol = 1e-10;
    unsigned int lagrange_max_iters = 20;
    double maier_saupe_alpha = 8.0;

    LiquidCrystalSystem<dim> lc_system(order,
                                       tria,
                                       degree,
                                       boundary_values_name,
                                       am,
                                       lagrange_step_size,
                                       lagrange_tol,
                                       lagrange_max_iters,
                                       maier_saupe_alpha);

    deserialize_lc_configuration(filename, lc_system);
}

#endif
