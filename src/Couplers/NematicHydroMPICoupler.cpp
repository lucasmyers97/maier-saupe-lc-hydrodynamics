#include "NematicHydroMPICoupler.hpp"

#include <deal.II/fe/fe_values.h>

template <int dim>
void NematicHydroMPICoupler<dim>::
assemble_hydro_system(NematicSystemMPI<dim> &ns,
                      HydroSystemMPI<dim> &hs)
{
    hs.system_matrix         = 0;
    hs.system_rhs            = 0;
    hs.preconditioner_matrix = 0;
    dealii::QGauss<dim> quadrature_formula(hs.fe.degree + 2);
    dealii::FEValues<dim> fe_values(hs.fe,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_JxW_values |
                                    dealii::update_gradients);
    dealii::FEValues<dim> ns_fe_values(ns.dof_handler.get_fe(),
                                       quadrature_formula,
                                       dealii::update_values |
                                       dealii::update_hessians);

    const unsigned int dofs_per_cell = hs.fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_Q_components = ns.dof_handler.get_fe().n_components();


    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                           dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    const dealii::FEValuesExtractors::Vector velocities(0);
    const dealii::FEValuesExtractors::Scalar pressure(dim);
    std::vector<dealii::SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<dealii::Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>                 div_phi_u(dofs_per_cell);
    std::vector<double>                 phi_p(dofs_per_cell);

    std::vector<dealii::Vector<double>>
        Q_vector_vals(n_q_points, dealii::Vector<double>(n_Q_components));
    std::vector<std::vector<dealii::Tensor<1, dim, double>>>
        Q_grad_vals(n_q_points,
                    std::vector<dealii::Tensor<1, dim, double>>
                    (n_Q_components));
    std::vector<dealii::Vector<double>>
        Q_laplace_vals(n_q_points, dealii::Vector<double>(n_Q_components));

    dealii::Tensor<2, dim, double> Q_mat;
    dealii::Vector<double> Lambda;
    dealii::SymmetricTensor<2, dim, double> H;
    dealii::SymmetricTensor<2, dim, double> sigma_d;

    auto cell = hs.dof_handler.begin_active();
    const auto endc = hs.dof_handler.end();
    auto lc_cell = ns.dof_handler.begin_active();

    for (; cell != endc; ++cell, ++lc_cell)
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            ns_fe_values.reinit(lc_cell);
            local_matrix                = 0;
            local_preconditioner_matrix = 0;
            local_rhs                   = 0;

            ns_fe_values.get_function_values(ns.current_solution,
                                             Q_vector_vals);
            ns_fe_values.get_function_gradients(ns.current_solution,
                                                Q_grad_vals);
            ns_fe_values.get_function_laplacians(ns.current_solution,
                                                 Q_laplace_vals);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                Lambda = 0;
                ns.lagrange_multiplier.invertQ(Q_vector_vals[q]);
                ns.lagrange_multiplier.returnLambda(Lambda);

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
                H[0][0] = (ns.maier_saupe_alpha * Q_vector_vals[q][0]
                           + Q_laplace_vals[q][0] - Lambda[0]);
                H[0][1] = (ns.maier_saupe_alpha * Q_vector_vals[q][1]
                           + Q_laplace_vals[q][1] - Lambda[1]);
                H[1][1] = (ns.maier_saupe_alpha * Q_vector_vals[q][3]
                           + Q_laplace_vals[q][3] - Lambda[3]);
                if (dim == 3)
                {
                    H[0][2] = (ns.maier_saupe_alpha * Q_vector_vals[q][2]
                               + Q_laplace_vals[q][2] - Lambda[2]);
                    H[1][2] = (ns.maier_saupe_alpha * Q_vector_vals[q][4]
                               + Q_laplace_vals[q][4] - Lambda[4]);
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
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        local_matrix(i, j) +=
                            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (1)
                             + hs.zeta_1 * dealii::scalar_product
                                           (symgrad_phi_u[i],
                                            Q_mat * symgrad_phi_u[j]
                                            - symgrad_phi_u[j] * Q_mat)
                             - div_phi_u[i] * phi_p[j]                 // (2)
                             - phi_p[i] * div_phi_u[j])                // (3)
                            * fe_values.JxW(q);                        // * dx

                        local_preconditioner_matrix(i, j) +=
                            (dealii::scalar_product(grad_phi_u[i],
                                                    grad_phi_u[j])
                            + phi_p[i] * phi_p[j]) // (4)
                            * fe_values.JxW(q);   // * dx
                    }

                    local_rhs(i) -= (dealii::
                                     scalar_product(grad_phi_u[i],
                                                    sigma_d)
                                     * hs.zeta_1
                                     * fe_values.JxW(q));
                    local_rhs(i) -= (dealii::scalar_product(
                                     grad_phi_u[i],
                                     H)
                                     * hs.zeta_2
                                     * fe_values.JxW(q));
                    local_rhs(i) -= (dealii::scalar_product(
                                     grad_phi_u[i],
                                     H * Q_mat - Q_mat * H)
                                     * hs.zeta_2
                                     * fe_values.JxW(q));
                }
            }

            cell->get_dof_indices(local_dof_indices);
            hs.constraints.distribute_local_to_global(local_matrix,
                                                      local_rhs,
                                                      local_dof_indices,
                                                      hs.system_matrix,
                                                      hs.system_rhs);
            hs.constraints.
                distribute_local_to_global(local_preconditioner_matrix,
                                           local_dof_indices,
                                           hs.preconditioner_matrix);
        }
    }
    hs.system_matrix.compress(dealii::VectorOperation::add);
    hs.preconditioner_matrix.compress(dealii::VectorOperation::add);
    hs.system_rhs.compress(dealii::VectorOperation::add);
}



// template <int dim>
// void NematicHydroMPICoupler<dim>::
// assemble_nematic_hydro_system(const NematicSystemMPI<dim> &nematic_system,
//                               HydroSystemMPI<dim> &hydro_system)
// {

// }

template class NematicHydroMPICoupler<2>;
template class NematicHydroMPICoupler<3>;
