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
                    Q_mat[2][2] = -(Q_mat[0][0] + Q_mat[1][1]);
                }

                sigma_d.clear();
                for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = i; j < dim; ++j)
                    {
                        for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                            sigma_d[i][j] -= (2 * Q_grad_vals[q][k][i]
                                              * Q_grad_vals[q][k][j]);

                        sigma_d[i][j] -= (Q_grad_vals[q][0][i]
                                          * Q_grad_vals[q][3][j]
                                          +
                                          Q_grad_vals[q][0][j]
                                          * Q_grad_vals[q][3][i]);
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
                            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (A)
                             - hs.eta_1 *
                             dealii::
                             scalar_product(grad_phi_u[i],
                                            symgrad_phi_u[j] * Q_mat -
                                            Q_mat * symgrad_phi_u[j])
                             - div_phi_u[i] * phi_p[j]    // (B^T)
                             - phi_p[i] * div_phi_u[j]) // (B)
                            * fe_values.JxW(q);      // * dx

                        local_preconditioner_matrix(i, j) +=
                            (dealii::
                             scalar_product(grad_phi_u[i], // (\approx A)
                                            grad_phi_u[j])
                             + phi_p[i] * phi_p[j])  // (M \approx B^T A B)
                            * fe_values.JxW(q); // * dx
                    }

                    local_rhs(i) -= ((dealii::
                                      scalar_product(grad_phi_u[i],
                                                     sigma_d)
                                      * hs.zeta_d)
                                     +
                                     (dealii::
                                      scalar_product(grad_phi_u[i],
                                                     H)
                                      * hs.zeta_2)
                                     +
                                     (dealii::
                                      scalar_product(grad_phi_u[i],
                                                     H * Q_mat - Q_mat * H)
                                      * hs.zeta_1))
                                     *
                                     fe_values.JxW(q); // * dx
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



template <int dim>
void NematicHydroMPICoupler<dim>::
assemble_nematic_hydro_system(NematicSystemMPI<dim> &ns,
                              HydroSystemMPI<dim> &hs,
                              double dt)
{
    hs.system_matrix         = 0;
    hs.system_rhs            = 0;
    hs.preconditioner_matrix = 0;

    ns.system_matrix = 0;
    ns.system_rhs = 0;

    dealii::QGauss<dim> quadrature_formula(hs.fe.degree + 2);
    dealii::FEValues<dim> hs_fe_values(hs.fe,
                                       quadrature_formula,
                                       dealii::update_values |
                                       dealii::update_JxW_values |
                                       dealii::update_gradients);
    dealii::FEValues<dim> ns_fe_values(ns.fe,
                                       quadrature_formula,
                                       dealii::update_values |
                                       dealii::update_JxW_values |
                                       dealii::update_gradients |
                                       dealii::update_hessians);

    const unsigned int hs_dofs_per_cell = hs.fe.n_dofs_per_cell();
    const unsigned int ns_dofs_per_cell = ns.fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_Q_components = ns.fe.n_components();

    dealii::FullMatrix<double> hs_local_matrix(hs_dofs_per_cell,
                                               hs_dofs_per_cell);
    dealii::FullMatrix<double> hs_local_preconditioner_matrix(hs_dofs_per_cell,
                                                              hs_dofs_per_cell);
    dealii::Vector<double> hs_local_rhs(hs_dofs_per_cell);
    std::vector<dealii::types::global_dof_index>
        hs_local_dof_indices(hs_dofs_per_cell);

    dealii::FullMatrix<double> ns_local_matrix(ns_dofs_per_cell,
                                               ns_dofs_per_cell);
    dealii::Vector<double> ns_local_rhs(ns_dofs_per_cell);
    std::vector<dealii::types::global_dof_index>
        ns_local_dof_indices(ns_dofs_per_cell);

    const dealii::FEValuesExtractors::Vector velocities(0);
    const dealii::FEValuesExtractors::Scalar pressure(dim);
    std::vector<dealii::SymmetricTensor<2, dim>>
        symgrad_phi_u(hs_dofs_per_cell);
    std::vector<dealii::Tensor<2, dim>> grad_phi_u(hs_dofs_per_cell);
    std::vector<double>                 div_phi_u(hs_dofs_per_cell);
    std::vector<double>                 phi_p(hs_dofs_per_cell);

    std::vector<dealii::Vector<double>>
        Q_vector_vals(n_q_points, dealii::Vector<double>(n_Q_components));
    std::vector<dealii::Vector<double>>
        old_Q_vector_vals(n_q_points, dealii::Vector<double>(n_Q_components));
    std::vector<std::vector<dealii::Tensor<1, dim, double>>>
        Q_grad_vals(n_q_points,
                    std::vector<dealii::Tensor<1, dim, double>>
                    (n_Q_components));
    std::vector<dealii::Vector<double>>
        Q_laplace_vals(n_q_points, dealii::Vector<double>(n_Q_components));

    dealii::Tensor<2, dim, double> Q_mat;
    dealii::SymmetricTensor<2, dim, double> H;
    dealii::SymmetricTensor<2, dim, double> sigma_d;

    dealii::Vector<double> Q(n_Q_components);
    dealii::Vector<double> Lambda(n_Q_components);;
    dealii::FullMatrix<double> R(ns.fe.components, ns.fe.components);
    std::vector<dealii::Vector<double>>
        R_inv_phi(ns_dofs_per_cell, dealii::Vector<double>(ns.fe.components));

    // data structures for factoring flow into Q-evolution
    std::vector<dealii::Tensor<1, dim>> u_vals(n_q_points);
    std::vector<dealii::Tensor<2, dim>> u_grads(n_q_points);
    std::vector<double> u_grad_phi(ns_dofs_per_cell);
    dealii::Vector<double> u_grad_Q(n_Q_components);

    dealii::Vector<double> W(3);
    dealii::Vector<double> eta_vec(n_Q_components);
    dealii::FullMatrix<double> eta_Jac(n_Q_components, n_Q_components);
    dealii::Vector<double> eps_vec(n_Q_components);
    std::vector<dealii::Vector<double>> eta_Jac_phi(ns_dofs_per_cell,
                                                    dealii::Vector<double>
                                                    (n_Q_components));
    dealii::Vector<double> tmp_vec(n_Q_components);

    auto ns_cell = ns.dof_handler.begin_active();
    const auto endc = ns.dof_handler.end();
    auto hs_cell = hs.dof_handler.begin_active();

    for (; ns_cell != endc; ++ns_cell, ++hs_cell)
    {
        if (ns_cell->is_locally_owned())
        {
            hs_fe_values.reinit(hs_cell);
            hs_local_matrix                = 0;
            hs_local_preconditioner_matrix = 0;
            hs_local_rhs                   = 0;

            ns_fe_values.reinit(ns_cell);
            ns_local_matrix                = 0;
            ns_local_rhs                   = 0;

            ns_fe_values.get_function_values(ns.current_solution,
                                             Q_vector_vals);
            ns_fe_values.get_function_values(ns.past_solution,
                                             old_Q_vector_vals);
            ns_fe_values.get_function_gradients(ns.current_solution,
                                                Q_grad_vals);
            ns_fe_values.get_function_laplacians(ns.current_solution,
                                                 Q_laplace_vals);

            hs_fe_values[velocities].get_function_values(hs.solution, u_vals);
            hs_fe_values[velocities].get_function_gradients(hs.solution, u_grads);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                Q = Q_vector_vals[q];
                Lambda = 0;
                R = 0;

                ns.lagrange_multiplier.invertQ(Q);
                ns.lagrange_multiplier.returnLambda(Lambda);
                ns.lagrange_multiplier.returnJac(R);
                for (unsigned int j = 0; j < ns_dofs_per_cell; ++j)
                {
                    const unsigned int component_j =
                        ns.fe.system_to_component_index(j).first;

                    R_inv_phi[j] = 0;
                    for (unsigned int i = 0; i < n_Q_components; ++i)
                        R_inv_phi[j][i] = (R(i, component_j)
                                           * ns_fe_values.shape_value(j, q));

                    tmp_vec.reinit(n_Q_components);
                    tmp_vec[component_j] = ns_fe_values.shape_value(j, q);
                    eta_Jac.vmult(eta_Jac_phi[j], tmp_vec);

                    u_grad_phi[j] = u_vals[q] * ns_fe_values.shape_grad(j, q);
                }

                Q_mat[0][0] = Q[0];
                Q_mat[0][1] = Q[1];
                Q_mat[1][1] = Q[3];
                Q_mat[1][0] = Q_mat[0][1];
                if (dim == 3)
                {
                    Q_mat[0][2] = Q[2];
                    Q_mat[1][2] = Q[4];
                    Q_mat[2][0] = Q_mat[0][2];
                    Q_mat[2][1] = Q_mat[1][2];
                }

                // Calculate u quadrature-specific values
                W[0] = 0.5 * (u_grads[q][1][0] - u_grads[q][0][1]);
                W[1] = (dim == 3) ? 0.5 * (u_grads[q][2][0] - u_grads[q][0][2]) : 0;
                W[2] = (dim == 3) ? 0.5 * (u_grads[q][2][1] - u_grads[q][1][2]) : 0;

                eta_vec[0] = -2 * (Q[1]*W[0] - Q[2]*W[1]);
                eta_vec[1] = Q[0]*W[0] - Q[2]*W[2] - Q[3]*W[0] - Q[4]*W[1];
                eta_vec[2] = 2*Q[0]*W[1] + Q[1]*W[2] + Q[3]*W[1] - Q[4]*W[0];
                eta_vec[3] = 2 * (Q[1]*W[0] - Q[4]*W[2]);
                eta_vec[4] = Q[0]*W[2] + Q[1]*W[1] + Q[2]*W[0] + 2*Q[3]*W[2];

                eta_Jac.reinit(n_Q_components, n_Q_components);
                eta_Jac[0][1] = -2*W[0];
                eta_Jac[0][2] = -2*W[1];
                eta_Jac[1][0] = W[0];
                eta_Jac[1][2] = -W[2];
                eta_Jac[1][3] = -W[0];
                eta_Jac[1][4] = -W[1];
                eta_Jac[2][0] = 2*W[1];
                eta_Jac[2][1] = W[2];
                eta_Jac[2][3] = W[1];
                eta_Jac[2][4] = -W[0];
                eta_Jac[3][1] = 2*W[0];
                eta_Jac[3][4] = -2*W[2];
                eta_Jac[4][0] = W[2];
                eta_Jac[4][1] = W[1];
                eta_Jac[4][2] = W[0];
                eta_Jac[4][3] = 2*W[2];

                eps_vec[0] = u_grads[q][0][0];
                eps_vec[1] = 0.5*(u_grads[q][1][0] + u_grads[q][0][1]);
                eps_vec[3] = u_grads[q][1][1];
                if (dim == 3)
                {
                    eps_vec[2] = 0.5 * (u_grads[q][2][0] + u_grads[q][0][2]);
                    eps_vec[4] = 0.5*(u_grads[q][2][1] + u_grads[q][1][2]);
                }

                for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                    u_grad_Q[k] = u_vals[q] * Q_grad_vals[q][k];

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


                for (unsigned int k = 0; k < hs_dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] =
                        hs_fe_values[velocities].symmetric_gradient(k, q);
                    grad_phi_u[k] =
                        hs_fe_values[velocities].gradient(k, q);
                    div_phi_u[k] = hs_fe_values[velocities].divergence(k, q);
                    phi_p[k]     = hs_fe_values[pressure].value(k, q);
                }

                for (unsigned int i = 0; i < hs_dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < hs_dofs_per_cell; ++j)
                    {
                        hs_local_matrix(i, j) +=
                            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (A)
                             - hs.eta_1 * dealii::scalar_product
                                           (symgrad_phi_u[i],
                                            symgrad_phi_u[j] * Q_mat
                                            - Q_mat * symgrad_phi_u[j])
                             - div_phi_u[i] * phi_p[j]                 // (B^T)
                             - phi_p[i] * div_phi_u[j])                // (B)
                            * hs_fe_values.JxW(q);                     // * dx

                        hs_local_preconditioner_matrix(i, j) +=
                            (dealii::
                             scalar_product(grad_phi_u[i],// (\approx A)
                                            grad_phi_u[j])
                             + phi_p[i] * phi_p[j])       // (M \approx B^T A B)
                            * hs_fe_values.JxW(q);        // * dx
                    }

                    hs_local_rhs(i) -= ((dealii::
                                         scalar_product(grad_phi_u[i],
                                                        sigma_d)
                                         * hs.zeta_d)
                                        +
                                        (dealii::
                                         scalar_product(grad_phi_u[i],
                                                        H)
                                         * hs.zeta_2)
                                        +
                                        (dealii::
                                         scalar_product(grad_phi_u[i],
                                                        H * Q_mat - Q_mat * H)
                                         * hs.zeta_1))
                                        *
                                        hs_fe_values.JxW(q); // * dx
                }

                for (unsigned int i = 0; i < ns_dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        ns.fe.system_to_component_index(i).first;

                    for (unsigned int j = 0; j < ns_dofs_per_cell; ++j)
                    {
                        const unsigned int component_j =
                            ns.fe.system_to_component_index(j).first;

                        ns_local_matrix(i, j) +=
                            (((component_i == component_j) ?
                              (ns_fe_values.shape_value(i, q)
                               * ns_fe_values.shape_value(j, q)) :
                              0)
                             +
                             ((component_i == component_j) ?
                              (dt
                               * ns_fe_values.shape_grad(i, q)
                               * ns_fe_values.shape_grad(j, q)) :
                              0)
                             +
                             (dt
                              * ns_fe_values.shape_value(i, q)
                              * R_inv_phi[j][component_i])
                             // +
                             // ((component_i == component_j) ?
                             //  (dt
                             //   * ns_fe_values.shape_value(i, q)
                             //   * u_grad_phi[j]) :
                             //  0)
                             // -
                             // (dt
                             //  * ns_fe_values.shape_value(i, q)
                             //  * eta_Jac_phi[j][component_i])
                             )
                            * ns_fe_values.JxW(q);
                    }
                    ns_local_rhs(i) +=
                        (-(ns_fe_values.shape_value(i, q)
                           * Q_vector_vals[q][component_i])
                         -
                         (dt
                          * ns_fe_values.shape_grad(i, q)
                          * Q_grad_vals[q][component_i])
                         -
                         (dt
                          * ns_fe_values.shape_value(i, q)
                          * Lambda[component_i])
                         +
                         ((1 + dt * ns.maier_saupe_alpha)
                          * ns_fe_values.shape_value(i, q)
                          * old_Q_vector_vals[q][component_i])
                         // -
                         // (dt
                         //  * ns_fe_values.shape_value(i, q)
                         //  * u_grad_Q[component_i])
                         // +
                         // (dt
                         //  * ns_fe_values.shape_value(i, q)
                         //  * eta_vec[component_i])
                         // -
                         // (dt
                         //  * ns_fe_values.shape_value(i, q)
                         //  * eps_vec[component_i]
                         //  * hs.gamma)
                         )
                        * ns_fe_values.JxW(q);
                }
            }

            ns_cell->get_dof_indices(ns_local_dof_indices);
            ns.constraints.distribute_local_to_global(ns_local_matrix,
                                                      ns_local_rhs,
                                                      ns_local_dof_indices,
                                                      ns.system_matrix,
                                                      ns.system_rhs);

            hs_cell->get_dof_indices(hs_local_dof_indices);
            hs.constraints.distribute_local_to_global(hs_local_matrix,
                                                      hs_local_rhs,
                                                      hs_local_dof_indices,
                                                      hs.system_matrix,
                                                      hs.system_rhs);
            hs.constraints.
                distribute_local_to_global(hs_local_preconditioner_matrix,
                                           hs_local_dof_indices,
                                           hs.preconditioner_matrix);
        }
    }
    hs.system_matrix.compress(dealii::VectorOperation::add);
    hs.preconditioner_matrix.compress(dealii::VectorOperation::add);
    hs.system_rhs.compress(dealii::VectorOperation::add);

    ns.system_matrix.compress(dealii::VectorOperation::add);
    ns.system_rhs.compress(dealii::VectorOperation::add);
}

template class NematicHydroMPICoupler<2>;
template class NematicHydroMPICoupler<3>;
