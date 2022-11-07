/**
 * This script creates a finite element object on a hypercube, then projects
 * the example function onto it.
 * Then it integrates the finite element function over the domain and
 * outputs the result.
 * The example function is just t * sin(x) * sin(y) which integrates over 
 * [0, pi] x [0, pi] to 4t (as one can check).
 * The purpose of this example is just to check that integration in deal.II
 * works the way that I expect it to.
 */

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <cmath>
#include <vector>

constexpr int dim = 2;

class ExampleFunction : public dealii::Function<dim>
{
public:
    ExampleFunction(double t_)
        : dealii::Function<dim>(1)
        , t(t_)
    {};

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        return t * std::sin(p[0]) * std::sin(p[1]);
    };

private:
    double t;
};

int main()
{
    const unsigned int degree = 1;
    const unsigned int n_refines = 8;
    const double left = 0.0;
    const double right = M_PI;

    const double t = 0.5;

    dealii::Triangulation<dim> tria;
    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(n_refines);

    dealii::DoFHandler<dim> dof_handler(tria);
    dealii::FE_Q<dim> fe(degree);
    dof_handler.distribute_dofs(fe);

    dealii::Vector<double> configuration(dof_handler.n_dofs());

    dealii::AffineConstraints<double> constraints;
    constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    dealii::VectorTools::project(dof_handler, 
                                 constraints, 
                                 dealii::QGauss<dim>(fe.degree + 1),
                                 ExampleFunction(t),
                                 configuration);


    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<double> function_vals(n_q_points);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    double domain_integral = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if ( !(cell->is_locally_owned()) )
            continue;

        cell->get_dof_indices(local_dof_indices);

        fe_values.reinit(cell);
        fe_values.get_function_values(configuration, function_vals);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            domain_integral += function_vals[q] * fe_values.JxW(q);
        }
    }

    std::cout << domain_integral << "\n";

    return 0;
}
