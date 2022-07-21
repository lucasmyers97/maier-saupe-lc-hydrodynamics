#ifndef HYDRO_FIXED_LC_DRIVER_HPP
#define HYDRO_FIXED_LC_DRIVER_HPP

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/parameter_handler.h>

#include <boost/archive/text_iarchive.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

#include "LiquidCrystalSystems/HydroFixedConfiguration.hpp"
#include "LiquidCrystalSystems/LiquidCrystalSystem.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"

template <int dim>
class HydroFixedLCDriver
{
public:
    HydroFixedLCDriver(){};
    void run();
    void declare_parameters();

private:
    void deserialize_lc_configuration(std::string filename,
                                      LiquidCrystalSystem<dim> &lc_system);
    void assemble_hydro_system(HydroFixedConfiguration<dim> &hydro_config,
                               LiquidCrystalSystem<dim> &lc_system);

    dealii::Triangulation<dim> tria;
    unsigned int degree;
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
void HydroFixedLCDriver<dim>::
assemble_hydro_system(HydroFixedConfiguration<dim> &hydro_config,
                      LiquidCrystalSystem<dim> &lc_system)
{
    const int order = 974;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const int max_iter = 20;
    LagrangeMultiplierAnalytic<dim> lagrange_multiplier(order,
                                                        alpha,
                                                        tol,
                                                        max_iter);

    const double maier_saupe_alpha = lc_system.return_parameters();

    int degree;
    double zeta_1;
    double zeta_2;

    std::tie(degree, zeta_1, zeta_2) = hydro_config.return_parameters();

    const dealii::DoFHandler<dim>& dof_handler
        = hydro_config.return_dof_handler();
    const dealii::FESystem<dim>& fe
        = hydro_config.return_fe();
    const dealii::AffineConstraints<double>& constraints
        = hydro_config.return_constraints();
    dealii::BlockSparseMatrix<double>& system_matrix
        = hydro_config.return_system_matrix();
    dealii::BlockVector<double>& system_rhs
        = hydro_config.return_system_rhs();
    dealii::BlockSparseMatrix<double>& preconditioner_matrix
        = hydro_config.return_preconditioner_matrix();

    const dealii::DoFHandler<dim>& lc_dof_handler
        = lc_system.return_dof_handler();
    const dealii::Vector<double>& lc_solution
        = lc_system.return_current_solution();

    system_matrix         = 0;
    system_rhs            = 0;
    preconditioner_matrix = 0;
    dealii::QGauss<dim> quadrature_formula(degree + 2);
    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_JxW_values |
                                    dealii::update_gradients);
    dealii::FEValues<dim> lc_fe_values(lc_dof_handler.get_fe(),
                                       quadrature_formula,
                                       dealii::update_values |
                                       dealii::update_hessians);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_Q_components = lc_dof_handler.get_fe().n_components();


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

    std::vector<dealii::Vector<double>>
        Q_vector_vals(n_q_points, dealii::Vector<double>(n_Q_components));
    std::vector<std::vector<dealii::Tensor<1, dim, double>>>
        Q_grad_vals(n_q_points,
                    std::vector<dealii::Tensor<1, dim, double>>(n_Q_components));
    std::vector<dealii::Vector<double>>
        Q_laplace_vals(n_q_points, dealii::Vector<double>(n_Q_components));

    dealii::Tensor<2, dim, double> Q_mat;
    dealii::Vector<double> Lambda;
    dealii::SymmetricTensor<2, dim, double> H;
    dealii::SymmetricTensor<2, dim, double> sigma_d;

    auto cell = dof_handler.begin_active();
    const auto endc = dof_handler.end();
    auto lc_cell = lc_dof_handler.begin_active();

    for (; cell != endc; ++cell, ++lc_cell)
    {
        fe_values.reinit(cell);
        lc_fe_values.reinit(lc_cell);
        local_matrix                = 0;
        local_preconditioner_matrix = 0;
        local_rhs                   = 0;

        lc_fe_values.get_function_values(lc_solution, Q_vector_vals);
        lc_fe_values.get_function_gradients(lc_solution, Q_grad_vals);
        lc_fe_values.get_function_laplacians(lc_solution, Q_laplace_vals);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            Lambda = 0;
            lagrange_multiplier.invertQ(Q_vector_vals[q]);
            lagrange_multiplier.returnLambda(Lambda);

            Q_mat[0][0] = Q_vector_vals[q][0];
            Q_mat[0][1] = Q_vector_vals[q][1];
            Q_mat[1][1] = Q_vector_vals[q][3];
            Q_mat[1][0] = Q_mat[0][1];
            if (dim == 3)
            {
              Q_mat[0][2] = Q_vector_vals[q][2];
              Q_mat[1][2] = Q_vector_vals[q][4];
              Q_mat[2][0] = Q_mat[0][2];
              Q_mat[2][1] = Q_mat[1][2];
            }

            sigma_d.clear();
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = i; j < dim; ++j)
                {
                    for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                        sigma_d[i][j] -= 2 * Q_grad_vals[q][k][i]
                                           * Q_grad_vals[q][k][j];

                    sigma_d[i][j] -= Q_grad_vals[q][0][i]
                                     * Q_grad_vals[q][3][j]
                                     +
                                     Q_grad_vals[q][0][j]
                                     * Q_grad_vals[q][3][i];
                }

            H.clear();
            H[0][0] = maier_saupe_alpha * Q_vector_vals[q][0]
                      + Q_laplace_vals[q][0] - Lambda[0];
            H[0][1] = maier_saupe_alpha * Q_vector_vals[q][1]
                      + Q_laplace_vals[q][1] - Lambda[1];
            H[1][1] = maier_saupe_alpha * Q_vector_vals[q][3]
                      + Q_laplace_vals[q][3] - Lambda[3];
            if (dim == 3)
            {
                H[0][2] = maier_saupe_alpha * Q_vector_vals[q][2]
                          + Q_laplace_vals[q][2] - Lambda[2];
                H[1][2] = maier_saupe_alpha * Q_vector_vals[q][4]
                          + Q_laplace_vals[q][4] - Lambda[4];
                H[2][2] = -(H[0][0] + H[1][1]);
            }


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
                                     Q_mat * symgrad_phi_u[j]
                                     - symgrad_phi_u[j] * Q_mat)
                         - div_phi_u[i] * phi_p[j]                 // (2)
                         - phi_p[i] * div_phi_u[j])                // (3)
                        * fe_values.JxW(q);                        // * dx

                    local_preconditioner_matrix(i, j) +=
                        (phi_p[i] * phi_p[j]) // (4)
                        * fe_values.JxW(q);   // * dx
                }

                // local_rhs(i) -= (dealii::scalar_product(
                //                  grad_phi_u[i],
                //                  sigma_d)
                //                  * zeta_1
                //                  * fe_values.JxW(q));
                // local_rhs(i) -= (dealii::scalar_product(
                //                  grad_phi_u[i],
                //                  H)
                //                  * zeta_2
                //                  * fe_values.JxW(q));
                local_rhs(i) -= (dealii::scalar_product(
                                 grad_phi_u[i],
                                 H * Q_mat - Q_mat * H)
                                 * zeta_2
                                 * fe_values.JxW(q));
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
void HydroFixedLCDriver<dim>::run()
{
    unsigned int degree = 1;
    std::string filename("two_defect_256_256.ar");

    LiquidCrystalSystem<dim> lc_system(tria, degree + 1);

    deserialize_lc_configuration(filename, lc_system);
    std::cout << "deserialization done\n";

    double zeta_1 = -0.9648241;
    double zeta_2 = 1.0050251;
    HydroFixedConfiguration<dim> hydro_config(tria, degree, zeta_1, zeta_2);

    hydro_config.setup_dofs();
    std::cout << "setting up dofs done\n";
    assemble_hydro_system(hydro_config, lc_system);
    std::cout << "assembling system done\n";
    hydro_config.solve_entire_block();
    std::cout << "solving done\n";
    hydro_config.output_results();
}

#endif
