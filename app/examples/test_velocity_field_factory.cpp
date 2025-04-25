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
        name = "quadratic"
        coupling-constant = 1.0
        endpoint_1 = [0.0, 0.0]
        endpoint_2 = [0.0, 1.0]
        flow_direction = [1.0, 0.0]
        max_flow_magnitude = 1.0
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
    unsigned int n_refines = 5;
    triangulation.refine_global(n_refines);
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

    std::ofstream file("quadratic_velocity_test.vtu");
 
    data_out.write_vtu(file);


    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_hessians
                                    | dealii::update_JxW_values);

    std::vector<dealii::Point<dim>> points;
    for (const auto &cell : triangulation.active_cell_iterators())
        for (std::size_t v = 0; v < cell->n_vertices(); ++v)
            points.push_back(cell->vertex(v));

    std::vector<std::vector<dealii::Tensor<1, dim, double>>> 
        gradients(points.size(),
                  std::vector<dealii::Tensor<1, dim>>(dim));

    v_field->vector_gradient_list(points, gradients);

    std::ofstream my_file;
    my_file.open("velocity_gradients.csv");
    if (dim == 3)
        my_file << "x,y,z,dvxdx,dvxdy,dvxdz,dvydx,dvydy,dvydz,dvzdx,dvzdy,dvzdz,\n";
    else if (dim == 2)
        my_file << "x,y,dvxdx,dvxdy,dvydx,dvydy,\n";

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        for (std::size_t j = 0; j < dim; ++j)
            my_file << points[i][j] << ",";

        for (std::size_t j = 0; j < dim; ++j)
            for (std::size_t k = 0; k < dim; ++k)
                my_file << gradients[i][j][k] << ",";

        my_file << "\n";
    }
    my_file.close();

    return 0;
}
