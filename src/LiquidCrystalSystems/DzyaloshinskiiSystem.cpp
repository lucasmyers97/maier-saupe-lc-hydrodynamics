#include "DzyaloshinskiiSystem.hpp"

#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/hdf5.h>

#include <cmath>
#include <fstream>
#include <ios>
#include <limits>

#include "Numerics/NumericalTools.hpp"

DzyaloshinskiiSystem::DzyaloshinskiiSystem(double eps_, unsigned int degree)
    : fe(degree)
    , dof_handler(tria)
    , eps(eps_)
{}



void DzyaloshinskiiSystem::make_grid(unsigned int n_refines)
{
    double left_endpoint = 0.0;
    double right_endpoint = 2 * M_PI;
    dealii::GridGenerator::hyper_cube(tria, left_endpoint, right_endpoint);
    tria.refine_global(n_refines);
}



void DzyaloshinskiiSystem::setup_system()
{
    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    {
        dealii::Table<2, double> polynomial_exponents(2, dim);
        std::vector<double> polynomial_coefficients(2);
        polynomial_exponents[0][0] = 1.0;
        polynomial_exponents[1][0] = 1.0;
        polynomial_coefficients[0] = 0;
        polynomial_coefficients[1] = 0.5;

        dealii::Functions::Polynomial<dim> 
            initial_configuration(polynomial_exponents, 
                                  polynomial_coefficients);

        constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, 
                                                        constraints);

        // right-hand boundary labeled with 1
        dealii::VectorTools::
            interpolate_boundary_values(dof_handler,
                                        0,
                                        initial_configuration,
                                        constraints);
        dealii::VectorTools::
            interpolate_boundary_values(dof_handler,
                                        1,
                                        initial_configuration,
                                        constraints);
        constraints.close();

        dealii::VectorTools::project(dof_handler,
                                     constraints,
                                     dealii::QGauss<dim>(fe.degree + 1),
                                     initial_configuration,
                                     solution);
    }


    constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // right-hand boundary labeled with a 1
    dealii::VectorTools::
        interpolate_boundary_values(dof_handler,
                                    0,
                                    dealii::Functions::ZeroFunction<dim>(1),
                                    constraints);
    dealii::VectorTools::
        interpolate_boundary_values(dof_handler,
                                    1,
                                    dealii::Functions::ZeroFunction<dim>(1),
                                    constraints);
    constraints.close();
 
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::
        make_sparsity_pattern(dof_handler,
                              dsp,
                              constraints,
                              /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

}



void DzyaloshinskiiSystem::assemble_system()
{
    system_matrix = 0;
    system_rhs = 0;

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_quadrature_points |
                                    dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);
           
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> 
        local_dof_indices(dofs_per_cell);

    std::vector<double> phi(n_q_points);
    std::vector<dealii::Tensor<1, dim>> dphi(n_q_points);
    std::vector<dealii::Point<dim>> quad_points(n_q_points);
    double theta;
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.get_function_values(solution, phi);
        fe_values.get_function_gradients(solution, dphi);
        quad_points = fe_values.get_quadrature_points();

        for (const auto q : fe_values.quadrature_point_indices())
        {
            theta = quad_points[q](0);

            for (const auto i : fe_values.dof_indices())
                for (const auto j : fe_values.dof_indices())
                    cell_matrix(i, j) -= 
                        (
                         ( (1 - eps * std::cos(2 * (phi[q] - theta)))
                            * fe_values.shape_grad(i, q)[0]
                            * fe_values.shape_grad(j, q)[0] )
                         + 
                         ( dphi[q][0] * 2 * eps * std::sin(2 * (phi[q] - theta))
                           * fe_values.shape_grad(i, q)[0]
                           * fe_values.shape_value(j, q) )
                         +
                         ( (4 * eps * std::cos(2 * (phi[q] - theta)) 
                            * dphi[q][0] * (dphi[q][0] - 1)
                            +
                            (2 * dphi[q][0] - dphi[q][0] * dphi[q][0])
                            * 2 * eps * std::cos(2 * (phi[q] - theta)) )
                           * fe_values.shape_value(i, q)
                           * fe_values.shape_value(j, q) )
                         )
                         * fe_values.JxW(q);

            for (const auto i : fe_values.dof_indices())
                cell_rhs(i) += 
                    (
                     ( fe_values.shape_grad(i, q)[0]
                       * (1 - eps * std::cos(2 * (phi[q] - theta)))
                       * dphi[q][0] )
                     +
                     ( fe_values.shape_value(i, q)
                       * dphi[q][0] * dphi[q][0]
                       * eps * std::sin(2 * (phi[q] - theta)) )
                    )
                    * fe_values.JxW(q);
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, 
                                               cell_rhs, 
                                               local_dof_indices, 
                                               system_matrix, 
                                               system_rhs);

    }
}



void DzyaloshinskiiSystem::solve_and_update(double newton_step)
{
    dealii::SparseDirectUMFPACK solver;
    solver.solve(system_matrix, system_rhs);
    constraints.distribute(system_rhs);

    solution.add(newton_step, system_rhs);
}



double DzyaloshinskiiSystem::
run_newton_method(double tol, unsigned int max_iters, double newton_step)
{
    double res = std::numeric_limits<double>::max();
    unsigned int iters = 0;

    while (res >= tol && iters < max_iters)
    {
        assemble_system();
        res = system_rhs.l2_norm();
        std::cout << "Step #: " << iters << "\n";
        std::cout << "Residual is: " << res << "\n\n";

        solve_and_update(newton_step);
        ++iters;
    }

    return res;
}



void DzyaloshinskiiSystem::output_solution(std::string filename)
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "dzyaloshinskii_solution");
    data_out.build_patches();

    std::ofstream output(filename);
    data_out.write_vtu(output);
}



void DzyaloshinskiiSystem::
output_hdf5(unsigned int n_points, std::string filename)
{
    /* Get phi evaluated at `n_points` theta values */
    const double left = 0.0;
    const double right = 2 * M_PI;

    std::vector<double> theta 
        = NumericalTools::linspace(left, right, n_points);
    std::vector<dealii::Point<dim>> points(n_points);
    for (std::size_t i = 0; i < points.size(); ++i)
        points[i][0] = theta[i];
    std::vector<double> phi(n_points);

    dealii::Functions::FEFieldFunction<dim> function(dof_handler, solution);
    function.value_list(points, phi);

    /* Write values to hdf5 file */
    dealii::HDF5::File 
        data_file(filename, dealii::HDF5::File::FileAccessMode::create);
    std::vector<hsize_t> dimensions = {n_points};
    auto theta_dataset = data_file.create_dataset<double>(std::string("theta"), 
                                                          dimensions);
    auto phi_dataset = data_file.create_dataset<double>(std::string("phi"), 
                                                        dimensions);
    theta_dataset.write(theta);
    phi_dataset.write(phi);
}
