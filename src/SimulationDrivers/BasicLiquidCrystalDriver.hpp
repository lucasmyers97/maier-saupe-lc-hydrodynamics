#ifndef BASIC_LIQUID_CRYSTAL_HPP
#define BASIC_LIQUID_CRYSTAL_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/parameter_handler.h>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <memory>
#include <tuple>
#include <string>
#include <iostream>
#include <fstream>

#include "LiquidCrystalSystems/LiquidCrystalSystem.hpp"

template <int dim>
class BasicLiquidCrystalDriver
{
public:
    BasicLiquidCrystalDriver(unsigned int num_refines_,
                             double left_,
                             double right_,
                             double dt_,
                             double simulation_tol_,
                             unsigned int simulation_max_iters_,
                             unsigned int n_steps_,
                             std::string data_folder_,
                             std::string config_filename_);

    void run();
    void declare_parameters(dealii::ParameterHandler &prm);
    void get_parameters(dealii::ParameterHandler &prm);

    void serialize_lc_system(LiquidCrystalSystem<dim> &lc_system);
    void deserialize_lc_system(LiquidCrystalSystem<dim> &lc_system);
    void run_deserialization();

private:
    void make_grid();
    void iterate_timestep(LiquidCrystalSystem<dim> &lc_system);

    dealii::Triangulation<dim> tria;
    unsigned int num_refines;
    double left;
    double right;
    double dt;

    double simulation_tol;
    unsigned int simulation_max_iters;
    unsigned int n_steps;
    std::string data_folder;
    std::string config_filename;
};



template <int dim>
BasicLiquidCrystalDriver<dim>::
    BasicLiquidCrystalDriver(unsigned int num_refines_,
                             double left_,
                             double right_,
                             double dt_,
                             double simulation_tol_,
                             unsigned int simulation_max_iters_,
                             unsigned int n_steps_,
                             std::string data_folder_,
                             std::string config_filename_)
    : num_refines(num_refines_)
    , left(left_)
    , right(right_)

    , dt(dt_)

    , simulation_tol(simulation_tol_)
    , simulation_max_iters(simulation_max_iters_)
    , n_steps(n_steps_)
    , data_folder(data_folder_)
    , config_filename(config_filename_)
{}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("BasicLiquidCrystalDriver");
    prm.declare_entry("Number of refines",
                      "6",
                      dealii::Patterns::Integer());
    prm.declare_entry("Left",
                      "-1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Right",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("dt",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Simulation tolerance",
                      "1e-10",
                      dealii::Patterns::Double());
    prm.declare_entry("Simulation maximum iterations",
                      "20",
                      dealii::Patterns::Integer());
    prm.declare_entry("Number of steps",
                      "30",
                      dealii::Patterns::Integer());
    prm.declare_entry("Data folder",
                      "./",
                      dealii::Patterns::DirectoryName());
    prm.declare_entry("Configuration filename",
                      "lc_configuration",
                      dealii::Patterns::FileName());
    prm.leave_subsection();
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
get_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("BasicLiquidCrystalDriver");
    num_refines = prm.get_integer("Number of refines");
    left = prm.get_double("Left");
    right = prm.get_double("Right");
    dt = prm.get_double("dt");
    simulation_tol = prm.get_double("Simulation tolerance");
    simulation_max_iters = prm.get_integer("Simulation maximum iterations");
    n_steps = prm.get_integer("Number of steps");
    data_folder = prm.get("Data folder");
    config_filename = prm.get("Configuration filename");
    prm.leave_subsection();
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::make_grid()
{
    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(num_refines);
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
    iterate_timestep(LiquidCrystalSystem<dim> &lc_system)
{
    lc_system.setup_system(/*initial_timestep = */false);

    unsigned int iterations = 0;
    double residual_norm{std::numeric_limits<double>::max()};
    while (residual_norm > simulation_tol && iterations < simulation_max_iters)
    {
        lc_system.assemble_system(dt);
        lc_system.solve();
        lc_system.update_current_solution(1.0);
        residual_norm = lc_system.return_norm();

        std::cout << "Residual norm is: " << residual_norm << "\n";
    }

    if (residual_norm > simulation_tol)
        std::terminate();

    lc_system.set_past_solution_to_current();
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::run()
{

    unsigned int degree = 1;
    std::string boundary_values_name = "two-defect";
    std::map<std::string, boost::any> am;
    am["S-value"] = 0.6751;
    am["defect-charge-name"] = std::string("plus-half-minus-half");
    am["centers"] = std::vector<double>({-35.0, 0, 35.0, 0});
    double maier_saupe_alpha = 8.0;
    int order = 974;
    double lagrange_step_size = 1.0;
    double lagrange_tol = 1e-10;
    unsigned int lagrange_max_iters = 20;

    make_grid();
    LiquidCrystalSystem<dim> lc_system(tria,
                                       degree,
                                       boundary_values_name,
                                       am,
                                       maier_saupe_alpha,
                                       order,
                                       lagrange_step_size,
                                       lagrange_tol,
                                       lagrange_max_iters);

    lc_system.setup_system(true);
    for (int current_step = 1; current_step < n_steps; ++current_step)
    {
        std::cout << "Starting timestep #" << current_step << "\n\n";

        iterate_timestep(lc_system);
        lc_system.output_results(data_folder, config_filename, current_step);

        std::cout << "Finished timestep\n\n";
    }

    serialize_lc_system(lc_system);

    const int new_order = 590;
    LiquidCrystalSystem<dim> new_lc_system(tria);
    deserialize_lc_system(new_lc_system);
    new_lc_system.output_results(data_folder, std::string("deserialized"), 1);
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::run_deserialization()
{
    LiquidCrystalSystem<dim> new_lc_system(tria);
    deserialize_lc_system(new_lc_system);
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::serialize_lc_system(
    LiquidCrystalSystem<dim> &lc_system)
{
    std::string filename("lc_system_archive.txt");
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << tria;
        oa << lc_system;
    }
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
deserialize_lc_system(LiquidCrystalSystem<dim> &lc_system)
{
    std::string filename("lc_system_archive.txt");
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> tria;
        ia >> lc_system;
    }
}



#endif
