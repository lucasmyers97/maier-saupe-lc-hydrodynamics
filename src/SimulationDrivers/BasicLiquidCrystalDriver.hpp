#ifndef BASIC_LIQUID_CRYSTAL_HPP
#define BASIC_LIQUID_CRYSTAL_HPP

#include <deal.II/grid/tria.h>

#include <deal.II/base/parameter_handler.h>

#include <string>

#include "LiquidCrystalSystems/LiquidCrystalSystem.hpp"

template <int dim>
class BasicLiquidCrystalDriver
{
public:
    BasicLiquidCrystalDriver(unsigned int degree_ = 1,
                             unsigned int num_refines_ = 6,
                             double left_ = 1.0,
                             double right_ = -1.0,
                             double dt_ = 1.0,
                             unsigned int n_steps_ = 1,
                             double simulation_tol_ = 1e-10,
                             unsigned int simulation_max_iters_ = 20,
                             std::string data_folder_ = std::string("./"),
                             std::string config_filename_ = std::string(""),
                             std::string archive_filename_
                             = std::string("lc_simulation.ar"));

    void run();
    static void declare_parameters(dealii::ParameterHandler &prm);
    void get_parameters(dealii::ParameterHandler &prm);

    void serialize_lc_system(LiquidCrystalSystem<dim> &lc_system,
                             std::string filename);
    void deserialize_lc_system(LiquidCrystalSystem<dim> &lc_system,
                               std::string filename);

private:
    void make_grid();
    void iterate_timestep(LiquidCrystalSystem<dim> &lc_system);

    dealii::Triangulation<dim> tria;
    unsigned int degree;
    unsigned int num_refines;
    double left;
    double right;

    double dt;
    unsigned int n_steps;

    double simulation_tol;
    unsigned int simulation_max_iters;

    std::string data_folder;
    std::string config_filename;
    std::string archive_filename;
};

#endif
