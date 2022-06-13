#ifndef BASIC_HYDRO_DRIVER_HPP
#define BASIC_HYDRO_DRIVER_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
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

#include <memory>
#include <tuple>

#include "LiquidCrystalSystems/HydroFixedConfiguration.hpp"

template <int dim>
class BasicHydroDriver
{
public:
    BasicHydroDriver(std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                     stress_tensor_,
                     std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                     Q_tensor_,
                     unsigned int num_refines_,
                     double left_,
                     double right_);

    void run();

private:
    void make_grid();
    void assemble_system(HydroFixedConfiguration<dim>&
                         hydro_fixed_configuration);

    dealii::Triangulation<dim> tria;
    std::unique_ptr<dealii::TensorFunction<2, dim, double>> stress_tensor;
    std::unique_ptr<dealii::TensorFunction<2, dim, double>> Q_tensor;
    unsigned int num_refines;
    double left;
    double right;

};



template <int dim>
BasicHydroDriver<dim>::
    BasicHydroDriver(std::unique_ptr<dealii::TensorFunction<2, dim, double>> stress_tensor_,
                     std::unique_ptr<dealii::TensorFunction<2, dim, double>> Q_tensor_,
                     unsigned int num_refines_,
                     double left_,
                     double right_)
    : stress_tensor(std::move(stress_tensor_))
    , Q_tensor(std::move(Q_tensor_))
    , num_refines(num_refines_)
    , left(left_)
    , right(right_)
{}


template <int dim>
void BasicHydroDriver<dim>::make_grid()
{
    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(num_refines);
}



template <int dim>
void BasicHydroDriver<dim>::
assemble_system(HydroFixedConfiguration<dim>& hydro_fixed_configuration)
{
    int degree;
    double zeta_1;
    double zeta_2;

    std::tie(degree, zeta_1, zeta_2)
        = hydro_fixed_configuration.return_parameters();

    const dealii::DoFHandler<dim>& dof_handler
        = hydro_fixed_configuration.return_dof_handler();
    const dealii::FESystem<dim>& fe
        = hydro_fixed_configuration.return_fe();
    const dealii::AffineConstraints<double>& constraints
        = hydro_fixed_configuration.return_constraints();
    dealii::BlockSparseMatrix<double>& system_matrix
        = hydro_fixed_configuration.return_system_matrix();
    dealii::BlockVector<double>& system_rhs
        = hydro_fixed_configuration.return_system_rhs();
    dealii::BlockSparseMatrix<double>& preconditioner_matrix
        = hydro_fixed_configuration.return_preconditioner_matrix();

    system_matrix         = 0;
    system_rhs            = 0;
    preconditioner_matrix = 0;
    dealii::QGauss<dim> quadrature_formula(degree + 2);
    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values |
                                    dealii::update_gradients);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();


    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                           dofs_per_cell);
    dealii::Vector<double>     local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const dealii::FEValuesExtractors::Vector velocities(0);
    const dealii::FEValuesExtractors::Scalar pressure(dim);
    std::vector<dealii::SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<dealii::Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    std::vector<dealii::Tensor<2, dim, double>> stress_tensor_vals(n_q_points);
    std::vector<dealii::Tensor<2, dim, double>> Q_tensor_vals(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix                = 0;
        local_preconditioner_matrix = 0;
        local_rhs                   = 0;

        stress_tensor->value_list(fe_values.get_quadrature_points(),
                                  stress_tensor_vals);
        Q_tensor->value_list(fe_values.get_quadrature_points(),
                             Q_tensor_vals);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                symgrad_phi_u[k] =
                    fe_values[velocities].symmetric_gradient(k, q);
                grad_phi_u[k] =
                    fe_values[velocities].gradient(k, q);
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                phi_p[k]     = fe_values[pressure].value(k, q);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j <= i; ++j)
                {
                    local_matrix(i, j) +=
                        (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (1)
                         + zeta_1 * dealii::scalar_product
                                    (symgrad_phi_u[i],
                                     Q_tensor_vals[q] * symgrad_phi_u[j]
                                     - symgrad_phi_u[j] * Q_tensor_vals[q])
                         - div_phi_u[i] * phi_p[j]                 // (2)
                         - phi_p[i] * div_phi_u[j])                // (3)
                        * fe_values.JxW(q);                        // * dx

                    local_preconditioner_matrix(i, j) +=
                        (phi_p[i] * phi_p[j]) // (4)
                        * fe_values.JxW(q);   // * dx
                }
                const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                local_rhs(i) -= (dealii::scalar_product(
                                 fe_values[velocities].gradient(i, q),
                                 stress_tensor_vals[q])
                                 * fe_values.JxW(q));
                local_rhs(i) -= (dealii::scalar_product(
                                 grad_phi_u[i],
                                 stress_tensor_vals[q] * Q_tensor_vals[q]
                                 - Q_tensor_vals[q] * stress_tensor_vals[q])
                                 * fe_values.JxW(q)
                                 * zeta_2);
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            {
                local_matrix(i, j) = local_matrix(j, i);
                local_preconditioner_matrix(i, j) =
                    local_preconditioner_matrix(j, i);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
        constraints.distribute_local_to_global(local_preconditioner_matrix,
                                               local_dof_indices,
                                               preconditioner_matrix);
    }
}



template <int dim>
void BasicHydroDriver<dim>::run()
{
    unsigned int degree = 1;
    double zeta_1 = 1.0;
    double zeta_2 = 1.0;

    make_grid();
    HydroFixedConfiguration<dim> hydro_fixed_config(degree, tria,
                                                    zeta_1, zeta_2);
    hydro_fixed_config.setup_dofs();
    // hydro_fixed_config.assemble_system(stress_tensor, Q_tensor);

    assemble_system(hydro_fixed_config);

    // hydro_fixed_config.solve();
    hydro_fixed_config.solve_entire_block();
    hydro_fixed_config.output_results();
}


#endif
