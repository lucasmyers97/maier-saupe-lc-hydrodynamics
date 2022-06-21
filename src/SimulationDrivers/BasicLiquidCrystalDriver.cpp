#include "SimulationDrivers/BasicLiquidCrystalDriver.hpp"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/grid/grid_generator.h>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <string>
#include <limits>
#include <exception>
#include <fstream>
#include <iostream>

template <int dim>
BasicLiquidCrystalDriver<dim>::
BasicLiquidCrystalDriver(unsigned int degree_,
                         unsigned int num_refines_,
                         double left_,
                         double right_,
                         double dt_,
                         unsigned int n_steps_,
                         double simulation_tol_,
                         unsigned int simulation_max_iters_,
                         std::string data_folder_,
                         std::string config_filename_,
                         std::string archive_filename_)
    : degree(degree_)
    , num_refines(num_refines_)
    , left(left_)
    , right(right_)

    , dt(dt_)
    , n_steps(n_steps_)

    , simulation_tol(simulation_tol_)
    , simulation_max_iters(simulation_max_iters_)

    , data_folder(data_folder_)
    , config_filename(config_filename_)
    , archive_filename(archive_filename_)
{}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("BasicLiquidCrystalDriver");

    prm.declare_entry("Finite element degree",
                      "1",
                      dealii::Patterns::Integer());
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
    prm.declare_entry("Number of steps",
                      "30",
                      dealii::Patterns::Integer());

    prm.declare_entry("Simulation tolerance",
                      "1e-10",
                      dealii::Patterns::Double());
    prm.declare_entry("Simulation maximum iterations",
                      "20",
                      dealii::Patterns::Integer());

    prm.declare_entry("Data folder",
                      "./",
                      dealii::Patterns::DirectoryName());
    prm.declare_entry("Configuration filename",
                      "lc_configuration",
                      dealii::Patterns::FileName());
    prm.declare_entry("Archive filename",
                      "lc_simulation.ar",
                      dealii::Patterns::FileName());

    prm.leave_subsection();
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
get_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("BasicLiquidCrystalDriver");

    degree = prm.get_integer("Finite element degree");
    num_refines = prm.get_integer("Number of refines");
    left = prm.get_double("Left");
    right = prm.get_double("Right");

    dt = prm.get_double("dt");
    n_steps = prm.get_integer("Number of steps");

    simulation_tol = prm.get_double("Simulation tolerance");
    simulation_max_iters = prm.get_integer("Simulation maximum iterations");

    data_folder = prm.get("Data folder");
    config_filename = prm.get("Configuration filename");
    archive_filename = prm.get("Archive filename");

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
    dealii::ParameterHandler prm;
    std::ifstream ifs("lc_parameters.prm");
    BasicLiquidCrystalDriver<dim>::declare_parameters(prm);
    LiquidCrystalSystem<dim>::declare_parameters(prm);
    prm.parse_input(ifs);
    get_parameters(prm);

    make_grid();

    LiquidCrystalSystem<dim> lc_system(tria, degree);
    lc_system.get_parameters(prm);

    lc_system.setup_system(true);
    lc_system.output_results(data_folder, config_filename, 0);
    for (int current_step = 1; current_step < n_steps; ++current_step)
    {
        std::cout << "Starting timestep #" << current_step << "\n\n";

        iterate_timestep(lc_system);
        lc_system.output_results(data_folder, config_filename, current_step);

        std::cout << "Finished timestep\n\n";
    }

    serialize_lc_system(lc_system, archive_filename);

    const int new_order = 590;
    LiquidCrystalSystem<dim> new_lc_system(tria);
    deserialize_lc_system(new_lc_system, archive_filename);
    new_lc_system.output_results(data_folder, std::string("deserialized"), 1);
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
serialize_lc_system(LiquidCrystalSystem<dim> &lc_system,
                    std::string filename)
{
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << degree;
        oa << tria;
        oa << lc_system;
    }
}



template <int dim>
void BasicLiquidCrystalDriver<dim>::
deserialize_lc_system(LiquidCrystalSystem<dim> &lc_system,
                      std::string filename)
{
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> degree;
        ia >> tria;
        ia >> lc_system;
    }
}

template class BasicLiquidCrystalDriver<2>;
template class BasicLiquidCrystalDriver<3>;
