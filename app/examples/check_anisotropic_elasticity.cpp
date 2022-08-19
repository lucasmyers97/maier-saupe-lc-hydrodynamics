#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/index_set.h>
#include <deal.II/numerics/data_out.h>

namespace LA = dealii::LinearAlgebraTrilinos;

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"

#include <deal.II/lac/affine_constraints.h>
#include <fstream>

template <int dim>
void assemble_R1(const dealii::DoFHandler<dim> &dof_handler,
                 const dealii::AffineConstraints<double> &constraints,
                 const LA::MPI::Vector &solution,
                 LA::MPI::Vector &system_rhs)
{
    const dealii::FESystem<dim> &fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(solution, dQ);
            fe_values.get_function_values(solution, Q_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    if (component_i == 0)
                        cell_rhs(i) +=
                            (
                             ((2*Q_vec[q][0] + Q_vec[q][3])
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 1)
                        cell_rhs(i) +=
                            (
                             (2*Q_vec[q][1]
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 2)
                        cell_rhs(i) +=
                            (
                             (2*Q_vec[q][2]
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 3)
                        cell_rhs(i) +=
                            (
                             ((Q_vec[q][0] + 2*Q_vec[q][3])
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 4)
                        cell_rhs(i) +=
                            (
                             (2*Q_vec[q][4]
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
        }
    }
    system_rhs.compress(dealii::VectorOperation::add);
}


template <int dim>
void assemble_E32(const dealii::DoFHandler<dim> &dof_handler,
                  const dealii::AffineConstraints<double> &constraints,
                  const LA::MPI::Vector &solution,
                  LA::MPI::Vector &system_rhs)
{
    const dealii::FESystem<dim> &fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(solution, dQ);
            fe_values.get_function_values(solution, Q_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    if (component_i == 0)
                        cell_rhs(i) +=
                            (
                             ((-dQ[q][0][0]*dQ[q][0][0] - dQ[q][0][0]*dQ[q][3][0] - dQ[q][1][0]*dQ[q][1][0] - dQ[q][2][0]*dQ[q][2][0] - dQ[q][3][0]*dQ[q][3][0] - dQ[q][4][0]*dQ[q][4][0])
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 1)
                        cell_rhs(i) +=
                            (
                             (-((dQ[q][0][0] + dQ[q][3][0])*(dQ[q][0][1] + dQ[q][3][1]) + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] + 2*dQ[q][4][0]*dQ[q][4][1])
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 2)
                        cell_rhs(i) +=
                            (
                             0
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 3)
                        cell_rhs(i) +=
                            (
                             ((-dQ[q][0][1]*dQ[q][0][1] - dQ[q][0][1]*dQ[q][3][1] - dQ[q][1][1]*dQ[q][1][1] - dQ[q][2][1]*dQ[q][2][1] - dQ[q][3][1]*dQ[q][3][1] - dQ[q][4][1]*dQ[q][4][1])
                              * fe_values.shape_value(i, q))
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 4)
                        cell_rhs(i) +=
                            (
                             0
                            )
                            * fe_values.JxW(q);                
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
        }
    }
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void assemble_E31(const dealii::DoFHandler<dim> &dof_handler,
                  const dealii::AffineConstraints<double> &constraints,
                  const LA::MPI::Vector &solution,
                  LA::MPI::Vector &system_rhs)
{
    const dealii::FESystem<dim> &fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(solution, dQ);
            fe_values.get_function_values(solution, Q_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    if (component_i == 0)
                        cell_rhs(i) +=
                            (
                             (-((dQ[q][0][0] 
                              + dQ[q][3][0])*Q_vec[q][0] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][1]) * fe_values.shape_grad(i, q)[0] 
                              - ((dQ[q][0][0] + dQ[q][3][0])*Q_vec[q][1] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][3]) * fe_values.shape_grad(i, q)[1] 
                              - (Q_vec[q][0]*dQ[q][0][0] + Q_vec[q][1]*dQ[q][0][1]) * fe_values.shape_grad(i, q)[0] 
                              - (Q_vec[q][1]*dQ[q][0][0] + Q_vec[q][3]*dQ[q][0][1]) * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 1)
                        cell_rhs(i) +=
                            (
                             (-2*(Q_vec[q][0]*dQ[q][1][0] 
                              + Q_vec[q][1]*dQ[q][1][1]) * fe_values.shape_grad(i, q)[0] 
                              - 2*(Q_vec[q][1]*dQ[q][1][0] + Q_vec[q][3]*dQ[q][1][1]) * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 2)
                        cell_rhs(i) +=
                            (
                             (-2*(Q_vec[q][0]*dQ[q][2][0] 
                              + Q_vec[q][1]*dQ[q][2][1]) * fe_values.shape_grad(i, q)[0] 
                              - 2*(Q_vec[q][1]*dQ[q][2][0] + Q_vec[q][3]*dQ[q][2][1]) * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 3)
                        cell_rhs(i) +=
                            (
                             (-((dQ[q][0][0] 
                              + dQ[q][3][0])*Q_vec[q][0] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][1]) * fe_values.shape_grad(i, q)[0] 
                              - ((dQ[q][0][0] + dQ[q][3][0])*Q_vec[q][1] + (dQ[q][0][1] + dQ[q][3][1])*Q_vec[q][3]) * fe_values.shape_grad(i, q)[1] 
                              - (Q_vec[q][0]*dQ[q][3][0] + Q_vec[q][1]*dQ[q][3][1]) * fe_values.shape_grad(i, q)[0] 
                              - (Q_vec[q][1]*dQ[q][3][0] + Q_vec[q][3]*dQ[q][3][1]) * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 4)
                        cell_rhs(i) +=
                            (
                             (-2*(Q_vec[q][0]*dQ[q][4][0] 
                              + Q_vec[q][1]*dQ[q][4][1]) * fe_values.shape_grad(i, q)[0] 
                              - 2*(Q_vec[q][1]*dQ[q][4][0] + Q_vec[q][3]*dQ[q][4][1]) * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q); 
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
        }
    }
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void assemble_E1(const dealii::DoFHandler<dim> &dof_handler,
                 const dealii::AffineConstraints<double> &constraints,
                 const LA::MPI::Vector &solution,
                 LA::MPI::Vector &system_rhs)
{
    const dealii::FESystem<dim> &fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(solution, dQ);
            fe_values.get_function_values(solution, Q_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    if (component_i == 0)
                        cell_rhs(i) +=
                            (
                             (-2*dQ[q][0][0]*fe_values.shape_grad(i, q)[0] 
                              - 2*dQ[q][0][1] * fe_values.shape_grad(i, q)[1] 
                              - dQ[q][3][0] * fe_values.shape_grad(i, q)[0] 
                              - dQ[q][3][1] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 1)
                        cell_rhs(i) +=
                            (
                             (-2*dQ[q][1][0]*fe_values.shape_grad(i, q)[0] 
                              - 2*dQ[q][1][1] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 2)
                        cell_rhs(i) +=
                            (
                             (-2*dQ[q][2][0]*fe_values.shape_grad(i, q)[0] 
                              - 2*dQ[q][2][1] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 3)
                        cell_rhs(i) +=
                            (
                             (-dQ[q][0][0]*fe_values.shape_grad(i, q)[0] 
                              - dQ[q][0][1] * fe_values.shape_grad(i, q)[1] 
                              - 2*dQ[q][3][0] * fe_values.shape_grad(i, q)[0] 
                              - 2*dQ[q][3][1] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 4)
                        cell_rhs(i) +=
                            (
                             (-2*dQ[q][4][0]*fe_values.shape_grad(i, q)[0] 
                              - 2*dQ[q][4][1] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);                
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
        }
    }
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void assemble_E2(const dealii::DoFHandler<dim> &dof_handler,
                 const dealii::AffineConstraints<double> &constraints,
                 const LA::MPI::Vector &solution,
                 LA::MPI::Vector &system_rhs)
{
    const dealii::FESystem<dim> &fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(solution, dQ);
            fe_values.get_function_values(solution, Q_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    if (component_i == 0)
                        cell_rhs(i) +=
                            (
                             (-dQ[q][0][0]*fe_values.shape_grad(i, q)[0] 
                              - dQ[q][1][0] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 1)
                        cell_rhs(i) +=
                            (
                             (-dQ[q][0][1]*fe_values.shape_grad(i, q)[0] 
                              - dQ[q][1][0] * fe_values.shape_grad(i, q)[0] 
                              - dQ[q][1][1] * fe_values.shape_grad(i, q)[1] 
                              - dQ[q][3][0] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 2)
                        cell_rhs(i) +=
                            (
                             (-dQ[q][2][0]*fe_values.shape_grad(i, q)[0] 
                              - dQ[q][4][0] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 3)
                        cell_rhs(i) +=
                            (
                             (-dQ[q][1][1]*fe_values.shape_grad(i, q)[0] 
                              - dQ[q][3][1] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q);
                    else if (component_i == 4)
                        cell_rhs(i) +=
                            (
                             (-dQ[q][2][1]*fe_values.shape_grad(i, q)[0] 
                              - dQ[q][4][1] * fe_values.shape_grad(i, q)[1])
                            )
                            * fe_values.JxW(q); 
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
        }
    }
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void assemble_dxQ2(const dealii::DoFHandler<dim> &dof_handler,
                   const dealii::AffineConstraints<double> &constraints,
                   const LA::MPI::Vector &solution,
                   LA::MPI::Vector &system_rhs)
{
    const dealii::FESystem<dim> &fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(solution, dQ);
            fe_values.get_function_values(solution, Q_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    if (component_i == 0)
                        cell_rhs(i) +=
                            (
                             dQ[q][1][0]*dQ[q][1][0]*fe_values.shape_value(i, q)
                            )
                            * fe_values.JxW(q);
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
        }
    }
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void assemble_d2xQ2(const dealii::DoFHandler<dim> &dof_handler,
                    const dealii::AffineConstraints<double> &constraints,
                    const LA::MPI::Vector &solution,
                    LA::MPI::Vector &system_rhs)
{
    const dealii::FESystem<dim> &fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_hessians
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<std::vector<dealii::Tensor<2, dim>>>
        ddQ(n_q_points,
            std::vector<dealii::Tensor<2, dim>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_hessians(solution, ddQ);
            fe_values.get_function_gradients(solution, dQ);
            fe_values.get_function_values(solution, Q_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    if (component_i == 0)
                        cell_rhs(i) +=
                            (
                             ddQ[q][1][0][0]*fe_values.shape_value(i, q)
                            )
                            * fe_values.JxW(q);
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
        }
    }
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void output_Q_components(const MPI_Comm &mpi_communicator,
                         const dealii::DoFHandler<dim> &dof_handler,
                         const LA::MPI::Vector &system_rhs,
                         const dealii::parallel::distributed::Triangulation<dim>
                         &triangulation,
                         const std::string folder,
                         const std::string filename,
                         const int time_step)
{
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    dealii::DataOut<dim> data_out;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> Q_names(msc::vec_dim<dim>);
    for (std::size_t i = 0; i < Q_names.size(); ++i)
        Q_names[i] = std::string("Q") + std::to_string(i);

    data_out.add_data_vector(system_rhs, Q_names);
    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    std::ofstream output(folder + filename + "_components"
                         + "_" + std::to_string(time_step)
                         + ".vtu");
    data_out.write_vtu_with_pvtu_record(folder, filename, time_step,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



int main(int ac, char* av[])
{
    try
    {
        if (ac - 1 != 1)
            throw std::invalid_argument("Error! Didn't input filename");
        std::string parameter_filename(av[1]);

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        const int dim = 2;
        const double left = -10.0;
        const double right = 10.0;
        const unsigned int num_refines = 9;
        const unsigned int degree = 2;

        dealii::ParameterHandler prm;

        NematicSystemMPI<dim>::declare_parameters(prm);
        prm.parse_input(parameter_filename, "", /*skip_undefined*/true, /*find_mandatory_entries*/true);

        dealii::parallel::distributed::Triangulation<dim> tria(mpi_communicator);
        dealii::GridGenerator::hyper_cube(tria, left, right);
        tria.refine_global(num_refines);

        NematicSystemMPI<dim> nematic_system(tria, degree);
        nematic_system.get_parameters(prm);

        nematic_system.setup_dofs(mpi_communicator, true);
        nematic_system.initialize_fe_field(mpi_communicator);
        nematic_system.output_Q_components(mpi_communicator, 
                                           tria, 
                                           std::string("./"), 
                                           std::string("test_periodic"), 
                                           0);

        const dealii::DoFHandler<dim> &dof_handler 
            = nematic_system.return_dof_handler();
        const dealii::AffineConstraints<double> &constraints 
            = nematic_system.return_constraints();
        const LA::MPI::Vector &solution
            = nematic_system.return_current_solution();

        LA::MPI::Vector R1;
        LA::MPI::Vector E1;
        LA::MPI::Vector E2;
        LA::MPI::Vector E31;
        LA::MPI::Vector E32;
        LA::MPI::Vector dxQ2;
        LA::MPI::Vector d2xQ2;

        d2xQ2.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
        dxQ2.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
        R1.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
        E1.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
        E2.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
        E31.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
        E32.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);

        assemble_d2xQ2(dof_handler, constraints, solution, d2xQ2);
        assemble_dxQ2(dof_handler, constraints, solution, dxQ2);
        assemble_R1(dof_handler, constraints, solution, R1);
        assemble_E1(dof_handler, constraints, solution, E1);
        assemble_E2(dof_handler, constraints, solution, E2);
        assemble_E31(dof_handler, constraints, solution, E31);
        assemble_E32(dof_handler, constraints, solution, E32);

        output_Q_components(mpi_communicator,
                            dof_handler,
                            d2xQ2,
                            tria,
                            std::string("./"),
                            std::string("d2xQ2_config"),
                            0);
        output_Q_components(mpi_communicator,
                            dof_handler,
                            dxQ2,
                            tria,
                            std::string("./"),
                            std::string("dxQ2_config"),
                            0);
        output_Q_components(mpi_communicator,
                            dof_handler,
                            R1,
                            tria,
                            std::string("./"),
                            std::string("R1_config"),
                            0);
        output_Q_components(mpi_communicator,
                            dof_handler,
                            E1,
                            tria,
                            std::string("./"),
                            std::string("E1_config"),
                            0);
        output_Q_components(mpi_communicator,
                            dof_handler,
                            E2,
                            tria,
                            std::string("./"),
                            std::string("E2_config"),
                            0);
        output_Q_components(mpi_communicator,
                            dof_handler,
                            E31,
                            tria,
                            std::string("./"),
                            std::string("E31_config"),
                            0);
        output_Q_components(mpi_communicator,
                            dof_handler,
                            E32,
                            tria,
                            std::string("./"),
                            std::string("E32_config"),
                            0);
        output_Q_components(mpi_communicator,
                            dof_handler,
                            solution,
                            tria,
                            std::string("./"),
                            std::string("Q_tensor_config"),
                            0);

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        return 1;
    }
}
