#ifndef BASIC_HYDRO_DRIVER_HPP
#define BASIC_HYDRO_DRIVER_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <memory>

#include "LiquidCrystalSystems/HydroFixedConfiguration.hpp"

template <int dim>
class BasicHydroDriver
{
public:
    BasicHydroDriver(std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                     stress_tensor_,
                     unsigned int num_refines);

    void run();

private:
    void make_grid();

    dealii::Triangulation<dim> tria;
    std::unique_ptr<dealii::TensorFunction<2, dim, double>> stress_tensor;
    unsigned int num_refines;

};



template <int dim>
BasicHydroDriver<dim>::
    BasicHydroDriver(std::unique_ptr<dealii::TensorFunction<2, dim, double>> stress_tensor_,
                     unsigned int num_refines_)
    : stress_tensor(std::move(stress_tensor_))
    , num_refines(num_refines_)
{}


template <int dim>
void BasicHydroDriver<dim>::make_grid()
{
    double left = -1.0;
    double right = 1.0;
    // unsigned int num_refines = 6;

    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(num_refines);
}


template <int dim>
void BasicHydroDriver<dim>::run()
{
    unsigned int degree = 1;

    make_grid();
    HydroFixedConfiguration<dim> hydro_fixed_config(degree, tria);
    hydro_fixed_config.setup_dofs();
    hydro_fixed_config.assemble_system(stress_tensor);
    // hydro_fixed_config.solve();
    hydro_fixed_config.solve_entire_block();
    hydro_fixed_config.output_results();
}


#endif
