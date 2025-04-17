#include "velocity_field/velocity_field.hpp"
#include "velocity_field/velocity_field_factory.hpp"
#include "deal.II/base/tensor.h"

#include <deal.II/base/point.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_generator.h>
#include <iostream>
#include <fstream>

int main()
{
    constexpr int dim = 2;
    static constexpr auto source = R"(
        name = "uniform"
        velocity_vector = [ 1.0, 1.0 ]
    )";

    auto vf_table = toml::parse(source);
    auto vf_params = VelocityFieldFactory::parse_parameters<dim>(vf_table);
    auto v_field = VelocityFieldFactory::velocity_field_factory<dim>(vf_params);

    dealii::Point<dim, double> p;
    dealii::Vector<double> value(dim);
    v_field->vector_value(p, value);
    std::cout << value << "\n";

    dealii::Triangulation<dim> triangulation;
    dealii::GridGenerator::hyper_cube(triangulation);
    const dealii::FESystem<dim> fe(dealii::FE_Q<dim>(1), dim);
    dealii::DoFHandler<dim> dof_handler(triangulation);
    triangulation.refine_global(5);
    dealii::Vector<double> solution;
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());

    dealii::VectorTools::interpolate(dof_handler, *v_field, solution);

    dealii::DataOut<dim> data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
  
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
  
    data_out.build_patches();

    std::ofstream file("uniform_velocity_test.vtu");
 
    data_out.write_vtu(file);

    return 0;
}
