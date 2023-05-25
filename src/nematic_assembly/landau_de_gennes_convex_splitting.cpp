#include "nematic_assembly/nematic_assembly.hpp"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include "Numerics/LagrangeMultiplierAnalytic.hpp"

namespace nematic_assembly {

namespace LA = dealii::LinearAlgebraTrilinos;

template <>
void landau_de_gennes_convex_splitting<2>(double dt, double A, double B, double C,
                                          double L2, double L3,
                                          const dealii::DoFHandler<2> &dof_handler,
                                          const LA::MPI::Vector &current_solution,
                                          const LA::MPI::Vector &past_solution,
                                          const dealii::AffineConstraints<double> &constraints,
                                          LA::MPI::SparseMatrix &system_matrix,
                                          LA::MPI::Vector &system_rhs)
{
    constexpr int dim = 2;

    const dealii::FESystem<dim> fe = dof_handler.get_fe();
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_matrix = 0;
    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if ( !(cell->is_locally_owned()) )
            continue;

        cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);

        fe_values.reinit(cell);
        fe_values.get_function_gradients(current_solution, dQ);
        fe_values.get_function_values(current_solution, Q_vec);
        fe_values.get_function_values(past_solution, Q0_vec);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i =
                    fe.system_to_component_index(i).first;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const unsigned int component_j =
                        fe.system_to_component_index(j).first;
                    if (component_i == 0 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (2*(A*dt + 1)
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-2*B*dt*Q_vec[q][3]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*C*dt*(6*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*Q_vec[q][0]*Q_vec[q][3] + 2*(Q_vec[q][1]) * (Q_vec[q][1]) + 2*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 2*(Q_vec[q][4]) * (Q_vec[q][4]))
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-dt*(
                                  - 2 * fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[0] 
                                  - 2 * fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[1]))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[0])
                                 +
                                 (L3*dt*(2*(Q_vec[q][1]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] + (2*dQ[q][0][0] + dQ[q][3][0])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[0] + ((dQ[q][0][0] + dQ[q][3][0])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][0] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][0] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][0][0]) * fe_values.shape_grad(i, q)[0]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[0])
                                 +
                                 (L3*dt*((2*dQ[q][0][0]*fe_values.shape_grad(i, q)[1] 
                                  + 2*dQ[q][0][1] * fe_values.shape_grad(i, q)[0] 
                                  + dQ[q][3][0] * fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][3][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q) + 2
                                  * fe_values.shape_value(i, q)*dQ[q][1][0]*fe_values.shape_grad(j, q)[0]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (4*C*dt*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][2][0]*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 ((A*dt + 1)
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-2*B*dt*(Q_vec[q][0] + Q_vec[q][3])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*C*dt*(3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*Q_vec[q][0]*Q_vec[q][3] + (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-dt*(
                                  - fe_values.shape_grad(i, q)[0] * fe_values.shape_grad(j, q)[0] 
                                  - fe_values.shape_grad(i, q)[1] * fe_values.shape_grad(j, q)[1]))
                                 +
                                 (L3*dt*((Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] + (dQ[q][0][0] + 2*dQ[q][3][0])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[0] + ((dQ[q][0][1] + dQ[q][3][1])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][0][1] * fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (-2*B*dt*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][4][0]*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[1])
                                 +
                                 (L3*dt*(((2*dQ[q][0][0] 
                                  + dQ[q][3][0]) * fe_values.shape_grad(j, q)[1] 
                                  + (2*dQ[q][0][1] + dQ[q][3][1]) * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q) + 2
                                  * fe_values.shape_value(j, q)*dQ[q][1][0]*fe_values.shape_grad(i, q)[0]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (2*(A*dt + 1)
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*B*dt*(Q_vec[q][0] + Q_vec[q][3])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*((Q_vec[q][0]) * (Q_vec[q][0]) + Q_vec[q][0]*Q_vec[q][3] + 3*(Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-dt*(
                                  - 2 * fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[0] 
                                  - 2 * fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[1]))
                                 +
                                 (-L2*dt*(
                                  - fe_values.shape_grad(i, q)[0] * fe_values.shape_grad(j, q)[0] 
                                  - fe_values.shape_grad(i, q)[1] * fe_values.shape_grad(j, q)[1]))
                                 +
                                 (2*L3*dt*((dQ[q][1][0]*fe_values.shape_grad(j, q)[1] 
                                  + dQ[q][1][1] * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q) 
                                  + (Q_vec[q][0] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][1][1]) * fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][1][0]) * fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (8*C*dt*Q_vec[q][1]*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt*(dQ[q][2][0]*fe_values.shape_grad(j, q)[1] 
                                  + dQ[q][2][1] * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[0])
                                 +
                                 (L3*dt*(((dQ[q][0][0] 
                                  + 2*dQ[q][3][0]) * fe_values.shape_grad(j, q)[1] 
                                  + (dQ[q][0][1] + 2*dQ[q][3][1]) * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q) + 2
                                  * fe_values.shape_value(j, q)*dQ[q][1][1]*fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (8*C*dt*Q_vec[q][1]*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt*(dQ[q][4][0]*fe_values.shape_grad(j, q)[1] 
                                  + dQ[q][4][1] * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (4*C*dt*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][2][0]*fe_values.shape_grad(i, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (8*C*dt*Q_vec[q][1]*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt*(dQ[q][2][0]*fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][2][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (2*(A*dt + 1)
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-2*B*dt*Q_vec[q][3]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*((Q_vec[q][0]) * (Q_vec[q][0]) + Q_vec[q][0]*Q_vec[q][3] + (Q_vec[q][1]) * (Q_vec[q][1]) + 3*(Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-dt*(
                                  - 2 * fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[0] 
                                  - 2 * fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[1]))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[0])
                                 +
                                 (2*L3*dt*((Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (-2*B*dt*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][2][1]*fe_values.shape_grad(i, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (8*C*dt*Q_vec[q][2]*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 ((A*dt + 1)
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-2*B*dt*(Q_vec[q][0] + Q_vec[q][3])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*C*dt*(3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*Q_vec[q][0]*Q_vec[q][3] + (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-dt*(
                                  - fe_values.shape_grad(i, q)[0] * fe_values.shape_grad(j, q)[0] 
                                  - fe_values.shape_grad(i, q)[1] * fe_values.shape_grad(j, q)[1]))
                                 +
                                 (L3*dt*((Q_vec[q][1]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] + (2*dQ[q][0][1] + dQ[q][3][1])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[1] + ((dQ[q][0][0] + dQ[q][3][0])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][0] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + fe_values.shape_value(j, q)*dQ[q][3][0] * fe_values.shape_grad(i, q)[0]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[1])
                                 +
                                 (L3*dt*((dQ[q][0][0]*fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][0][1] * fe_values.shape_grad(i, q)[0] 
                                  + 2*dQ[q][3][0] * fe_values.shape_grad(i, q)[1] 
                                  + 2*dQ[q][3][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q) + 2
                                  * fe_values.shape_value(i, q)*dQ[q][1][1]*fe_values.shape_grad(j, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (-2*B*dt*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][2][1]*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (2*(A*dt + 1)
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-2*B*dt*Q_vec[q][0]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*C*dt*(3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*Q_vec[q][0]*Q_vec[q][3] + 2*(Q_vec[q][1]) * (Q_vec[q][1]) + 2*(Q_vec[q][2]) * (Q_vec[q][2]) + 6*(Q_vec[q][3]) * (Q_vec[q][3]) + 2*(Q_vec[q][4]) * (Q_vec[q][4]))
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-dt*(
                                  - 2 * fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[0] 
                                  - 2 * fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[1]))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[1])
                                 +
                                 (L3*dt*(2*(Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] + (dQ[q][0][1] + 2*dQ[q][3][1])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[1] + ((dQ[q][0][1] + dQ[q][3][1])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][3][1]) * fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (4*C*dt*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][4][1]*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (-2*B*dt*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][4][0]*fe_values.shape_grad(i, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (8*C*dt*Q_vec[q][1]*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt*(dQ[q][4][0]*fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][4][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (2*B*dt*Q_vec[q][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (8*C*dt*Q_vec[q][2]*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (4*C*dt*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][4][1]*fe_values.shape_grad(i, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (2*(A*dt + 1)
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-2*B*dt*Q_vec[q][0]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (4*C*dt*((Q_vec[q][0]) * (Q_vec[q][0]) + Q_vec[q][0]*Q_vec[q][3] + (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + 3*(Q_vec[q][4]) * (Q_vec[q][4]))
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (-dt*(
                                  - 2 * fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[0] 
                                  - 2 * fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[1]))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[1])
                                 +
                                 (2*L3*dt*((Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);     
                }
                if (component_i == 0)
                    cell_rhs(i) +=
                        (
                         ((-(A*dt + 1)*(2*Q_vec[q][0] + Q_vec[q][3]) + 2*Q0_vec[q][0] + Q0_vec[q][3])
                          * fe_values.shape_value(i, q))
                         +
                         (B*dt*(2*Q_vec[q][0]*Q_vec[q][3] - (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))
                          * fe_values.shape_value(i, q))
                         +
                         (-C*dt*(2*Q_vec[q][0] + Q_vec[q][3])*((Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + (Q_vec[q][0]) * (Q_vec[q][0]) + 2*(Q_vec[q][1]) * (Q_vec[q][1]) + 2*(Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + 2*(Q_vec[q][4]) * (Q_vec[q][4]))
                          * fe_values.shape_value(i, q))
                         +
                         (-dt*(2*dQ[q][0][0]*fe_values.shape_grad(i, q)[0] 
                          + 2*dQ[q][0][1] * fe_values.shape_grad(i, q)[1] 
                          + dQ[q][3][0] * fe_values.shape_grad(i, q)[0] 
                          + dQ[q][3][1] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L2*dt*(dQ[q][0][0]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][1][0] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L3*dt*(((dQ[q][0][0] 
                          + dQ[q][3][0])*Q_vec[q][0] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][1]) * fe_values.shape_grad(i, q)[0] 
                          + ((dQ[q][0][0] + dQ[q][3][0])*Q_vec[q][1] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][3]) * fe_values.shape_grad(i, q)[1] 
                          + (Q_vec[q][0]*dQ[q][0][0] + Q_vec[q][1]*dQ[q][0][1]) * fe_values.shape_grad(i, q)[0] 
                          + (Q_vec[q][1]*dQ[q][0][0] + Q_vec[q][3]*dQ[q][0][1]) * fe_values.shape_grad(i, q)[1] + ((dQ[q][0][0]) * (dQ[q][0][0]) + dQ[q][0][0]*dQ[q][3][0] + (dQ[q][1][0]) * (dQ[q][1][0]) + (dQ[q][2][0]) * (dQ[q][2][0]) + (dQ[q][3][0]) * (dQ[q][3][0]) + (dQ[q][4][0]) * (dQ[q][4][0]))
                          * fe_values.shape_value(i, q)))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 1)
                    cell_rhs(i) +=
                        (
                         (-2*((A*dt + 1)*Q_vec[q][1] - Q0_vec[q][1])
                          * fe_values.shape_value(i, q))
                         +
                         (-2*B*dt*(Q_vec[q][0]*Q_vec[q][1] + Q_vec[q][1]*Q_vec[q][3] + Q_vec[q][2]*Q_vec[q][4])
                          * fe_values.shape_value(i, q))
                         +
                         (-4*C*dt*((Q_vec[q][0]) * (Q_vec[q][0]) + Q_vec[q][0]*Q_vec[q][3] + (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][1]
                          * fe_values.shape_value(i, q))
                         +
                         (-2*dt*(dQ[q][1][0]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][1][1] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L2*dt*(dQ[q][0][1]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][1][0] * fe_values.shape_grad(i, q)[0] 
                          + dQ[q][1][1] * fe_values.shape_grad(i, q)[1] 
                          + dQ[q][3][0] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L3*dt*(2*(Q_vec[q][0]*dQ[q][1][0] 
                          + Q_vec[q][1]*dQ[q][1][1]) * fe_values.shape_grad(i, q)[0] 
                          + 2*(Q_vec[q][1]*dQ[q][1][0] + Q_vec[q][3]*dQ[q][1][1]) * fe_values.shape_grad(i, q)[1] + ((dQ[q][0][0] + dQ[q][3][0])*(dQ[q][0][1] + dQ[q][3][1]) + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] + 2*dQ[q][4][0]*dQ[q][4][1])
                          * fe_values.shape_value(i, q)))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 2)
                    cell_rhs(i) +=
                        (
                         (-2*((A*dt + 1)*Q_vec[q][2] - Q0_vec[q][2])
                          * fe_values.shape_value(i, q))
                         +
                         (2*B*dt*(-Q_vec[q][1]*Q_vec[q][4] + Q_vec[q][2]*Q_vec[q][3])
                          * fe_values.shape_value(i, q))
                         +
                         (-4*C*dt*((Q_vec[q][0]) * (Q_vec[q][0]) + Q_vec[q][0]*Q_vec[q][3] + (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][2]
                          * fe_values.shape_value(i, q))
                         +
                         (-2*dt*(dQ[q][2][0]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][2][1] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L2*dt*(dQ[q][2][0]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][4][0] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-2*L3*dt*((Q_vec[q][0]*dQ[q][2][0] 
                          + Q_vec[q][1]*dQ[q][2][1]) * fe_values.shape_grad(i, q)[0] 
                          + (Q_vec[q][1]*dQ[q][2][0] + Q_vec[q][3]*dQ[q][2][1]) * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 3)
                    cell_rhs(i) +=
                        (
                         ((-(A*dt + 1)*(Q_vec[q][0] + 2*Q_vec[q][3]) + Q0_vec[q][0] + 2*Q0_vec[q][3])
                          * fe_values.shape_value(i, q))
                         +
                         (B*dt*((Q_vec[q][0]) * (Q_vec[q][0]) + 2*Q_vec[q][0]*Q_vec[q][3] - (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]))
                          * fe_values.shape_value(i, q))
                         +
                         (-C*dt*(Q_vec[q][0] + 2*Q_vec[q][3])*((Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + (Q_vec[q][0]) * (Q_vec[q][0]) + 2*(Q_vec[q][1]) * (Q_vec[q][1]) + 2*(Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + 2*(Q_vec[q][4]) * (Q_vec[q][4]))
                          * fe_values.shape_value(i, q))
                         +
                         (-dt*(dQ[q][0][0]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][0][1] * fe_values.shape_grad(i, q)[1] 
                          + 2*dQ[q][3][0] * fe_values.shape_grad(i, q)[0] 
                          + 2*dQ[q][3][1] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L2*dt*(dQ[q][1][1]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][3][1] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L3*dt*(((dQ[q][0][0] 
                          + dQ[q][3][0])*Q_vec[q][0] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][1]) * fe_values.shape_grad(i, q)[0] 
                          + ((dQ[q][0][0] + dQ[q][3][0])*Q_vec[q][1] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][3]) * fe_values.shape_grad(i, q)[1] 
                          + (Q_vec[q][0]*dQ[q][3][0] + Q_vec[q][1]*dQ[q][3][1]) * fe_values.shape_grad(i, q)[0] 
                          + (Q_vec[q][1]*dQ[q][3][0] + Q_vec[q][3]*dQ[q][3][1]) * fe_values.shape_grad(i, q)[1] + ((dQ[q][0][1]) * (dQ[q][0][1]) + dQ[q][0][1]*dQ[q][3][1] + (dQ[q][1][1]) * (dQ[q][1][1]) + (dQ[q][2][1]) * (dQ[q][2][1]) + (dQ[q][3][1]) * (dQ[q][3][1]) + (dQ[q][4][1]) * (dQ[q][4][1]))
                          * fe_values.shape_value(i, q)))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 4)
                    cell_rhs(i) +=
                        (
                         (-2*((A*dt + 1)*Q_vec[q][4] - Q0_vec[q][4])
                          * fe_values.shape_value(i, q))
                         +
                         (2*B*dt*(Q_vec[q][0]*Q_vec[q][4] - Q_vec[q][1]*Q_vec[q][2])
                          * fe_values.shape_value(i, q))
                         +
                         (-4*C*dt*((Q_vec[q][0]) * (Q_vec[q][0]) + Q_vec[q][0]*Q_vec[q][3] + (Q_vec[q][1]) * (Q_vec[q][1]) + (Q_vec[q][2]) * (Q_vec[q][2]) + (Q_vec[q][3]) * (Q_vec[q][3]) + (Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][4]
                          * fe_values.shape_value(i, q))
                         +
                         (-2*dt*(dQ[q][4][0]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][4][1] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-L2*dt*(dQ[q][2][1]*fe_values.shape_grad(i, q)[0] 
                          + dQ[q][4][1] * fe_values.shape_grad(i, q)[1]))
                         +
                         (-2*L3*dt*((Q_vec[q][0]*dQ[q][4][0] 
                          + Q_vec[q][1]*dQ[q][4][1]) * fe_values.shape_grad(i, q)[0] 
                          + (Q_vec[q][1]*dQ[q][4][0] + Q_vec[q][3]*dQ[q][4][1]) * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);            
            }
        }
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
    }
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}



template <>
void landau_de_gennes_convex_splitting<3>(double dt, double A, double B, double C,
                                          double L2, double L3,
                                          const dealii::DoFHandler<3> &dof_handler,
                                          const LA::MPI::Vector &current_solution,
                                          const LA::MPI::Vector &past_solution,
                                          const dealii::AffineConstraints<double> &constraints,
                                          LA::MPI::SparseMatrix &system_matrix,
                                          LA::MPI::Vector &system_rhs)
{
    throw std::logic_error("landau_de_gennes_convex_splitting<3> not implemented yet");
}

} // nematic_assembly