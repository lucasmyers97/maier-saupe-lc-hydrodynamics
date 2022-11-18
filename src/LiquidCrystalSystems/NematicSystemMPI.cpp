#include "NematicSystemMPI.hpp"

#include <deal.II/distributed/tria.h>
#include <deal.II/base/hdf5.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/differentiation/ad/ad_helpers.h>

#include <boost/program_options.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/any.hpp>

#include "Utilities/Output.hpp"
#include "Utilities/maier_saupe_constants.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Postprocessors/DirectorPostprocessor.hpp"
#include "Postprocessors/SValuePostprocessor.hpp"
#include "Postprocessors/EvaluateFEObject.hpp"
#include "Postprocessors/NematicPostprocessor.hpp"
#include "Postprocessors/EnergyPostprocessor.hpp"
#include "Postprocessors/ConfigurationForcePostprocessor.hpp"
#include "Numerics/FindDefects.hpp"

#include <deal.II/numerics/vector_tools_boundary.h>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <utility>
#include <tuple>

namespace
{
    namespace msc = maier_saupe_constants;
}



template <int dim>
NematicSystemMPI<dim>::
NematicSystemMPI(const dealii::parallel::distributed::Triangulation<dim>
                 &triangulation,
                 unsigned int degree,
                 std::string boundary_values_name,
                 const std::map<std::string, boost::any> &am,
                 double maier_saupe_alpha_,
                 double L2_,
                 double L3_,
                 int order,
                 double lagrange_step_size,
                 double lagrange_tol,
                 unsigned int lagrange_max_iters)
    : dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(degree), msc::vec_dim<dim>)
    , boundary_value_func(BoundaryValuesFactory::
                          BoundaryValuesFactory<dim>(am))
    , lagrange_multiplier(order,
                          lagrange_step_size,
                          lagrange_tol,
                          lagrange_max_iters)

    , maier_saupe_alpha(maier_saupe_alpha_)
    , L2(L2_)
    , L3(L3_)

    , defect_pts(/* time + dim + charge = */ dim + 2)
    , energy_vals(/* time + number of energy terms + squared energy = */ 7)
{}



template <int dim>
void NematicSystemMPI<dim>::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Nematic system MPI");
    BoundaryValuesFactory::declare_parameters<dim>(prm);

    prm.declare_entry("Maier saupe alpha",
                      "8.0",
                      dealii::Patterns::Double());
    prm.declare_entry("L2",
                      "0.0",
                      dealii::Patterns::Double());
    prm.declare_entry("L3",
                      "0.0",
                      dealii::Patterns::Double());

    prm.declare_entry("Lebedev order",
                      "590",
                      dealii::Patterns::Integer());
    prm.declare_entry("Lagrange step size",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Lagrange tolerance",
                      "1e-10",
                      dealii::Patterns::Double());
    prm.declare_entry("Lagrange maximum iterations",
                      "20",
                      dealii::Patterns::Integer());
    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPI<dim>::get_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Nematic system MPI");

    auto bv_params = BoundaryValuesFactory::get_parameters<dim>(prm);
    boundary_value_func = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(bv_params);

    maier_saupe_alpha = prm.get_double("Maier saupe alpha");
    L2 = prm.get_double("L2");
    L3 = prm.get_double("L3");

    int order = prm.get_integer("Lebedev order");
    double lagrange_step_size = prm.get_double("Lagrange step size");
    double lagrange_tol = prm.get_double("Lagrange tolerance");
    int lagrange_max_iter = prm.get_integer("Lagrange maximum iterations");

    lagrange_multiplier = LagrangeMultiplierAnalytic<dim>(order, 
                                                          lagrange_step_size, 
                                                          lagrange_tol, 
                                                          lagrange_max_iter);

    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPI<dim>::setup_dofs(const MPI_Comm &mpi_communicator,
                                       const bool initial_step)
{
    if (initial_step)
    {
        dof_handler.distribute_dofs(fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                        locally_relevant_dofs);

        // make constraints for system update
        constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                        constraints);

        if (boundary_value_func->boundary_condition 
            == std::string("Dirichlet"))
        {
            dealii::VectorTools::
                interpolate_boundary_values(dof_handler,
                                            /* boundary_component = */0,
                                            dealii::Functions::
                                            ZeroFunction<dim>(msc::vec_dim<dim>),
                                            constraints);
        }
        constraints.close();
    }
    // make sparsity pattern based on constraints
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                            dsp,
                                            constraints,
                                            /*keep_constrained_dofs=*/false);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                       locally_owned_dofs,
                                                       mpi_communicator,
                                                       locally_relevant_dofs);
    constraints.condense(dsp);

    system_rhs.reinit(locally_owned_dofs,
                      mpi_communicator);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    system_matrix.compress(dealii::VectorOperation::insert);
    system_rhs.compress(dealii::VectorOperation::insert);
}



template <int dim>
void NematicSystemMPI<dim>::
initialize_fe_field(const MPI_Comm &mpi_communicator)
{
    // impose boundary conditions on initial condition
    dealii::AffineConstraints<double> configuration_constraints;
    configuration_constraints.clear();
    configuration_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::
        make_hanging_node_constraints(dof_handler,
                                      configuration_constraints);
    dealii::VectorTools::
        interpolate_boundary_values(dof_handler,
                                    /* boundary_component = */0,
                                    *boundary_value_func,
                                    configuration_constraints);
    configuration_constraints.close();

    // interpolate initial condition
    LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                           mpi_communicator);
    dealii::VectorTools::interpolate(dof_handler,
                                     *boundary_value_func,
                                     locally_owned_solution);
    configuration_constraints.distribute(locally_owned_solution);
    locally_owned_solution.compress(dealii::VectorOperation::insert);

    // write completely distributed solution to current and past solutions
    current_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    past_solution.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);
    current_solution = locally_owned_solution;
    past_solution = locally_owned_solution;

    current_solution.compress(dealii::VectorOperation::insert);
    past_solution.compress(dealii::VectorOperation::insert);
}



template <int dim>
void NematicSystemMPI<dim>::assemble_system(const double dt)
{
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
        old_solution_gradients
        (n_q_points,
         std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        old_solution_values(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        previous_solution_values(n_q_points,
                                 dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda(fe.components);
    dealii::FullMatrix<double> R(fe.components, fe.components);
    std::vector<dealii::Vector<double>>
        R_inv_phi(dofs_per_cell, dealii::Vector<double>(fe.components));

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(current_solution,
                                             old_solution_gradients);
            fe_values.get_function_values(current_solution,
                                          old_solution_values);
            fe_values.get_function_values(past_solution,
                                          previous_solution_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                Lambda = 0;
                R = 0;

                lagrange_multiplier.invertQ(old_solution_values[q]);
                lagrange_multiplier.returnLambda(Lambda);
                lagrange_multiplier.returnJac(R);
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const unsigned int component_j =
                        fe.system_to_component_index(j).first;

                    R_inv_phi[j] = 0;
                    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
                        R_inv_phi[j][i] = (R(i, component_j)
                                           * fe_values.shape_value(j, q));
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int component_j =
                            fe.system_to_component_index(j).first;

                        cell_matrix(i, j) +=
                            (((component_i == component_j) ?
                              (fe_values.shape_value(i, q)
                               * fe_values.shape_value(j, q)) :
                              0)
                             +
                             ((component_i == component_j) ?
                              (dt
                               * fe_values.shape_grad(i, q)
                               * fe_values.shape_grad(j, q)) :
                              0)
                             +
                             (dt
                              * fe_values.shape_value(i, q)
                              * R_inv_phi[j][component_i]))
                            * fe_values.JxW(q);
                    }
                    cell_rhs(i) +=
                        (-(fe_values.shape_value(i, q)
                           * old_solution_values[q][component_i])
                         -
                         (dt
                          * fe_values.shape_grad(i, q)
                          * old_solution_gradients[q][component_i])
                         -
                         (dt
                          * fe_values.shape_value(i, q)
                          * Lambda[component_i])
                         +
                         ((1 + dt * maier_saupe_alpha)
                          * fe_values.shape_value(i, q)
                          * previous_solution_values[q][component_i])
                         )
                        * fe_values.JxW(q);
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
        }
    }
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void NematicSystemMPI<dim>::
assemble_system_anisotropic(double dt)
{
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

    dealii::Vector<double> Lambda_vec(fe.components);
    dealii::FullMatrix<double> dLambda_dQ(fe.components, fe.components);

    const double alpha = maier_saupe_alpha;
    // const double L2 = 0;
    // const double L3 = 3.0;

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
            Lambda_vec = 0;
            dLambda_dQ = 0;

            lagrange_multiplier.invertQ(Q_vec[q]);
            lagrange_multiplier.returnLambda(Lambda_vec);
            lagrange_multiplier.returnJac(dLambda_dQ);

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
                                 (2
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (dt*(2*dLambda_dQ[0][0] + dLambda_dQ[3][0])
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
                                 (dt*(2*dLambda_dQ[0][1] + dLambda_dQ[3][1])
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
                                 (dt*(2*dLambda_dQ[0][2] + dLambda_dQ[3][2])
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
                                 (fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (dt*(2*dLambda_dQ[0][3] + dLambda_dQ[3][3])
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
                                 (dt*(2*dLambda_dQ[0][4] + dLambda_dQ[3][4])
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
                                 (2*dt*dLambda_dQ[1][0]
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
                                 (2
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*dt*dLambda_dQ[1][1]
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
                                 (2*dt*dLambda_dQ[1][2]
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
                                 (2*dt*dLambda_dQ[1][3]
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
                                 (2*dt*dLambda_dQ[1][4]
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
                                 (2*dt*dLambda_dQ[2][0]
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
                                 (2*dt*dLambda_dQ[2][1]
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
                                 (2
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*dt*dLambda_dQ[2][2]
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
                                 (2*dt*dLambda_dQ[2][3]
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
                                 (2*dt*dLambda_dQ[2][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[1]*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (dt*(dLambda_dQ[0][0] + 2*dLambda_dQ[3][0])
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
                                 (dt*(dLambda_dQ[0][1] + 2*dLambda_dQ[3][1])
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
                                 (dt*(dLambda_dQ[0][2] + 2*dLambda_dQ[3][2])
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
                                 (2
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (dt*(dLambda_dQ[0][3] + 2*dLambda_dQ[3][3])
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
                                 (dt*(dLambda_dQ[0][4] + 2*dLambda_dQ[3][4])
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
                                 (2*dt*dLambda_dQ[4][0]
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
                                 (2*dt*dLambda_dQ[4][1]
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
                                 (2*dt*dLambda_dQ[4][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (L2*dt*fe_values.shape_grad(i, q)[0]*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[4][3]
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
                                 (2
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                 +
                                 (2*dt*dLambda_dQ[4][4]
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
                         (-(2*Q_vec[q][0] + Q_vec[q][3])
                          * fe_values.shape_value(i, q))
                         +
                         ((alpha * dt + 1)
                          *(2*Q0_vec[q][0] + Q0_vec[q][3])
                          * fe_values.shape_value(i, q))
                         +
                         (-dt*(2*Lambda_vec[0] + Lambda_vec[3])
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
                         (-2*Q_vec[q][1]
                          * fe_values.shape_value(i, q))
                         +
                         (2*(alpha * dt + 1)
                          *Q0_vec[q][1]
                          * fe_values.shape_value(i, q))
                         +
                         (-2*dt*Lambda_vec[1]
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
                         (-2*Q_vec[q][2]
                          * fe_values.shape_value(i, q))
                         +
                         (2*(alpha * dt + 1)
                          *Q0_vec[q][2]
                          * fe_values.shape_value(i, q))
                         +
                         (-2*dt*Lambda_vec[2]
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
                         (-(Q_vec[q][0] + 2*Q_vec[q][3])
                          * fe_values.shape_value(i, q))
                         +
                         ((alpha * dt + 1)
                          *(Q0_vec[q][0] + 2*Q0_vec[q][3])
                          * fe_values.shape_value(i, q))
                         +
                         (-dt*(Lambda_vec[0] + 2*Lambda_vec[3])
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
                         (-2*Q_vec[q][4]
                          * fe_values.shape_value(i, q))
                         +
                         (2*(alpha * dt + 1)
                          *Q0_vec[q][4]
                          * fe_values.shape_value(i, q))
                         +
                         (-2*dt*Lambda_vec[4]
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



template <int dim>
void NematicSystemMPI<dim>::solve_and_update(const MPI_Comm &mpi_communicator,
                                             const double alpha)
{
    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-10);
    LA::SolverGMRES solver(solver_control);
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix);

    LA::MPI::Vector completely_distributed_update(locally_owned_dofs,
                                                  mpi_communicator);
    solver.solve(system_matrix,
                 completely_distributed_update,
                 system_rhs,
                 preconditioner);
    constraints.distribute(completely_distributed_update);

    // update current_solution -- must transfer to completely distributed vector
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    completely_distributed_solution = current_solution;
    completely_distributed_solution.add(alpha, completely_distributed_update);
    current_solution = completely_distributed_solution;
}



template <int dim>
double NematicSystemMPI<dim>::return_norm()
{
    return system_rhs.l2_norm();
}



template <int dim>
double NematicSystemMPI<dim>::return_linfty_norm()
{
    return system_rhs.linfty_norm();
}



template <int dim>
void NematicSystemMPI<dim>::
set_past_solution_to_current(const MPI_Comm &mpi_communicator)
{
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    completely_distributed_solution = current_solution;
    past_solution = completely_distributed_solution;
}



template <int dim>
void NematicSystemMPI<dim>::
find_defects(double min_dist, 
             double charge_threshold, 
             double current_time)
{
    std::vector<dealii::Point<dim>> local_minima;
    std::vector<double> defect_charges;
    std::tie(local_minima, defect_charges) 
        = NumericalTools::find_defects(dof_handler, 
                                       current_solution, 
                                       min_dist, 
                                       charge_threshold);
    for (const auto &pt : local_minima)
    {
        defect_pts[0].push_back(current_time);
        defect_pts[1].push_back(pt[0]);
        defect_pts[2].push_back(pt[1]);
        if (dim == 3)
            defect_pts[3].push_back(pt[2]);
    }

    for (const auto &charge : defect_charges)
        defect_pts[dim + 1].push_back(charge);
}



template <int dim>
void NematicSystemMPI<dim>::
calc_energy(const MPI_Comm &mpi_communicator, double current_time)
{
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_hessians
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<std::vector<dealii::Tensor<2, dim>>>
        ddQ(n_q_points,
            std::vector<dealii::Tensor<2, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    double Z = 0;

    const double alpha = maier_saupe_alpha;

    double mean_field_term = 0;
    double entropy_term = 0;
    double L1_elastic_term = 0;
    double L2_elastic_term = 0;
    double L3_elastic_term = 0;

    double dE_dQ_squared = 0;

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if ( !(cell->is_locally_owned()) )
            continue;

        cell->get_dof_indices(local_dof_indices);

        fe_values.reinit(cell);
        fe_values.get_function_gradients(current_solution, dQ);
        fe_values.get_function_hessians(current_solution, ddQ);
        fe_values.get_function_values(current_solution, Q_vec);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            Lambda_vec = 0;

            lagrange_multiplier.invertQ(Q_vec[q]);
            lagrange_multiplier.returnLambda(Lambda_vec);
            Z = lagrange_multiplier.returnZ();

            mean_field_term += 
                (alpha*(-(Q_vec[q][0]) * (Q_vec[q][0]) 
                        - Q_vec[q][0]*Q_vec[q][3] 
                        - (Q_vec[q][1]) * (Q_vec[q][1]) 
                        - (Q_vec[q][2]) * (Q_vec[q][2]) 
                        - (Q_vec[q][3]) * (Q_vec[q][3]) 
                        - (Q_vec[q][4]) * (Q_vec[q][4])))
                * fe_values.JxW(q);
            
            entropy_term +=
                (2*Q_vec[q][0]*Lambda_vec[0] + Q_vec[q][0]*Lambda_vec[3] 
                 + 2*Q_vec[q][1]*Lambda_vec[1] + 2*Q_vec[q][2]*Lambda_vec[2] 
                 + Q_vec[q][3]*Lambda_vec[0] + 2*Q_vec[q][3]*Lambda_vec[3] 
                 + 2*Q_vec[q][4]*Lambda_vec[4] - std::log(Z) + std::log(4*M_PI))
                * fe_values.JxW(q);
            
            L1_elastic_term +=
                ((1.0/2.0)*(-dQ[q][0][0] - dQ[q][3][0]) 
                 * (-dQ[q][0][0] - dQ[q][3][0]) 
                 + (1.0/2.0)*(-dQ[q][0][1] - dQ[q][3][1]) 
                 * (-dQ[q][0][1] - dQ[q][3][1]) 
                 + (1.0/2.0)*(dQ[q][0][0]) 
                 * (dQ[q][0][0]) 
                 + (1.0/2.0)*(dQ[q][0][1]) * (dQ[q][0][1]) 
                 + (dQ[q][1][0]) * (dQ[q][1][0]) 
                 + (dQ[q][1][1]) * (dQ[q][1][1]) 
                 + (dQ[q][2][0]) * (dQ[q][2][0]) 
                 + (dQ[q][2][1]) * (dQ[q][2][1]) 
                 + (1.0/2.0)*(dQ[q][3][0]) * (dQ[q][3][0]) 
                 + (1.0/2.0)*(dQ[q][3][1]) * (dQ[q][3][1]) 
                 + (dQ[q][4][0]) * (dQ[q][4][0]) 
                 + (dQ[q][4][1]) * (dQ[q][4][1]))
                * fe_values.JxW(q);
            
            L2_elastic_term +=
                ((1.0/2.0)*L2
                 * ((dQ[q][0][0] + dQ[q][1][1]) * (dQ[q][0][0] + dQ[q][1][1]) 
                 + (dQ[q][1][0] + dQ[q][3][1]) * (dQ[q][1][0] + dQ[q][3][1]) 
                 + (dQ[q][2][0] + dQ[q][4][1]) * (dQ[q][2][0] + dQ[q][4][1])))
                * fe_values.JxW(q);
            
            L3_elastic_term +=
                ((1.0/2.0)*L3
                 *(2*((-dQ[q][0][0] - dQ[q][3][0])*(-dQ[q][0][1] - dQ[q][3][1]) 
                         + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] 
                         + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] 
                         + 2*dQ[q][4][0]*dQ[q][4][1])*Q_vec[q][1] 
                     + ((-dQ[q][0][0] - dQ[q][3][0]) 
                         * (-dQ[q][0][0] - dQ[q][3][0]) 
                         + (dQ[q][0][0]) * (dQ[q][0][0]) 
                         + 2*(dQ[q][1][0]) * (dQ[q][1][0]) 
                         + 2*(dQ[q][2][0]) * (dQ[q][2][0]) 
                         + (dQ[q][3][0]) * (dQ[q][3][0]) 
                         + 2*(dQ[q][4][0]) * (dQ[q][4][0]))*Q_vec[q][0] 
                     + ((-dQ[q][0][1] - dQ[q][3][1]) 
                         * (-dQ[q][0][1] - dQ[q][3][1]) 
                         + (dQ[q][0][1]) * (dQ[q][0][1]) 
                         + 2*(dQ[q][1][1]) * (dQ[q][1][1]) 
                         + 2*(dQ[q][2][1]) * (dQ[q][2][1]) 
                         + (dQ[q][3][1]) * (dQ[q][3][1]) 
                         + 2*(dQ[q][4][1]) * (dQ[q][4][1]))*Q_vec[q][3]))
                  * fe_values.JxW(q);

            dE_dQ_squared += (
                                (1.0/2.0)*std::pow(L2*(ddQ[q][2][0][0] 
                                + ddQ[q][4][0][1]) 
                                + 2*L3*(Q_vec[q][0]*ddQ[q][2][0][0] 
                                + 2*Q_vec[q][1]*ddQ[q][2][0][1] 
                                + Q_vec[q][3]*ddQ[q][2][1][1] 
                                + dQ[q][0][0]*dQ[q][2][0] 
                                + dQ[q][1][0]*dQ[q][2][1] 
                                + dQ[q][1][1]*dQ[q][2][0] 
                                + dQ[q][2][1]*dQ[q][3][1]) 
                                + 2*alpha*Q_vec[q][2] 
                                - 2*Lambda_vec[2] 
                                + 2*ddQ[q][2][0][0] 
                                + 2*ddQ[q][2][1][1], 2) 
                                + (1.0/2.0)*std::pow(L2*(ddQ[q][4][1][1] 
                                + ddQ[q][2][0][1]) 
                                + 2*L3*(Q_vec[q][0]*ddQ[q][4][0][0] 
                                + 2*Q_vec[q][1]*ddQ[q][4][0][1] 
                                + Q_vec[q][3]*ddQ[q][4][1][1] 
                                + dQ[q][0][0]*dQ[q][4][0] 
                                + dQ[q][1][0]*dQ[q][4][1] 
                                + dQ[q][1][1]*dQ[q][4][0] 
                                + dQ[q][3][1]*dQ[q][4][1]) 
                                + 2*alpha*Q_vec[q][4] 
                                - 2*Lambda_vec[4] 
                                + 2*ddQ[q][4][0][0] 
                                + 2*ddQ[q][4][1][1], 2) 
                                + (1.0/9.0)*std::pow(L2*(
                                -ddQ[q][0][0][0] 
                                + 2*ddQ[q][3][1][1] 
                                + ddQ[q][1][0][1]) 
                                + L3*(3*Q_vec[q][0]*ddQ[q][3][0][0] 
                                + 6*Q_vec[q][1]*ddQ[q][3][0][1] 
                                + 3*Q_vec[q][3]*ddQ[q][3][1][1] 
                                + std::pow(dQ[q][0][0], 2) 
                                + 4*dQ[q][0][0]*dQ[q][3][0] 
                                - 2*std::pow(dQ[q][0][1], 2) 
                                - 2*dQ[q][0][1]*dQ[q][3][1] 
                                + std::pow(dQ[q][1][0], 2) 
                                + 3*dQ[q][1][0]*dQ[q][3][1] 
                                - 2*std::pow(dQ[q][1][1], 2) 
                                + 3*dQ[q][1][1]*dQ[q][3][0] 
                                + std::pow(dQ[q][2][0], 2) 
                                - 2*std::pow(dQ[q][2][1], 2) 
                                + std::pow(dQ[q][3][0], 2) 
                                + std::pow(dQ[q][3][1], 2) 
                                + std::pow(dQ[q][4][0], 2) 
                                - 2*std::pow(dQ[q][4][1], 2)) 
                                + 3*alpha*Q_vec[q][3] 
                                - 3*Lambda_vec[3] 
                                + 3*ddQ[q][3][0][0] 
                                + 3*ddQ[q][3][1][1], 2) 
                                + (1.0/9.0)*std::pow(L2*(2*ddQ[q][0][0][0] 
                                - ddQ[q][3][1][1] 
                                + ddQ[q][1][0][1]) 
                                + L3*(3*Q_vec[q][0]*ddQ[q][0][0][0] 
                                + 6*Q_vec[q][1]*ddQ[q][0][0][1] 
                                + 3*Q_vec[q][3]*ddQ[q][0][1][1] 
                                + std::pow(dQ[q][0][0], 2) 
                                + 3*dQ[q][0][0]*dQ[q][1][1] 
                                - 2*dQ[q][0][0]*dQ[q][3][0] 
                                + std::pow(dQ[q][0][1], 2) 
                                + 3*dQ[q][0][1]*dQ[q][1][0] 
                                + 4*dQ[q][0][1]*dQ[q][3][1] 
                                - 2*std::pow(dQ[q][1][0], 2) 
                                + std::pow(dQ[q][1][1], 2) 
                                - 2*std::pow(dQ[q][2][0], 2) 
                                + std::pow(dQ[q][2][1], 2) 
                                - 2*std::pow(dQ[q][3][0], 2) 
                                + std::pow(dQ[q][3][1], 2) 
                                - 2*std::pow(dQ[q][4][0], 2) 
                                + std::pow(dQ[q][4][1], 2)) 
                                + 3*alpha*Q_vec[q][0] 
                                - 3*Lambda_vec[0] 
                                + 3*ddQ[q][0][0][0] 
                                + 3*ddQ[q][0][1][1], 2) 
                                + (1.0/2.0)*std::pow(L2*(ddQ[q][1][0][0] 
                                + ddQ[q][1][1][1] 
                                + ddQ[q][0][0][1] 
                                + ddQ[q][3][0][1]) 
                                + L3*(
                                -(dQ[q][0][0] 
                                + dQ[q][3][0])*(dQ[q][0][1] 
                                + dQ[q][3][1]) 
                                + 2*Q_vec[q][0]*ddQ[q][1][0][0] 
                                + 4*Q_vec[q][1]*ddQ[q][1][0][1] 
                                + 2*Q_vec[q][3]*ddQ[q][1][1][1] 
                                - dQ[q][0][0]*dQ[q][0][1] 
                                + 2*dQ[q][0][0]*dQ[q][1][0] 
                                + 2*dQ[q][1][0]*dQ[q][1][1] 
                                + 2*dQ[q][1][1]*dQ[q][3][1] 
                                - 2*dQ[q][2][0]*dQ[q][2][1] 
                                - dQ[q][3][0]*dQ[q][3][1] 
                                - 2*dQ[q][4][0]*dQ[q][4][1]) 
                                + 2*alpha*Q_vec[q][1] 
                                - 2*Lambda_vec[1] 
                                + 2*ddQ[q][1][0][0] 
                                + 2*ddQ[q][1][1][1], 2) 
                                + (1.0/9.0)*std::pow(L2*(ddQ[q][0][0][0] 
                                + ddQ[q][3][1][1] 
                                + 2*ddQ[q][1][0][1]) 
                                + L3*(3*Q_vec[q][0]*ddQ[q][0][0][0] 
                                + 3*Q_vec[q][0]*ddQ[q][3][0][0] 
                                + 6*Q_vec[q][1]*ddQ[q][0][0][1] 
                                + 6*Q_vec[q][1]*ddQ[q][3][0][1] 
                                + 3*Q_vec[q][3]*ddQ[q][0][1][1] 
                                + 3*Q_vec[q][3]*ddQ[q][3][1][1] 
                                + 2*std::pow(dQ[q][0][0], 2) 
                                + 3*dQ[q][0][0]*dQ[q][1][1] 
                                + 2*dQ[q][0][0]*dQ[q][3][0] 
                                - std::pow(dQ[q][0][1], 2) 
                                + 3*dQ[q][0][1]*dQ[q][1][0] 
                                + 2*dQ[q][0][1]*dQ[q][3][1] 
                                - std::pow(dQ[q][1][0], 2) 
                                + 3*dQ[q][1][0]*dQ[q][3][1] 
                                - std::pow(dQ[q][1][1], 2) 
                                + 3*dQ[q][1][1]*dQ[q][3][0] 
                                - std::pow(dQ[q][2][0], 2) 
                                - std::pow(dQ[q][2][1], 2) 
                                - std::pow(dQ[q][3][0], 2) 
                                + 2*std::pow(dQ[q][3][1], 2) 
                                - std::pow(dQ[q][4][0], 2) 
                                - std::pow(dQ[q][4][1], 2)) 
                                + 3*alpha*(Q_vec[q][0] 
                                + Q_vec[q][3]) 
                                - 3*Lambda_vec[0] 
                                - 3*Lambda_vec[3] 
                                + 3*ddQ[q][0][0][0] 
                                + 3*ddQ[q][0][1][1] 
                                + 3*ddQ[q][3][0][0] 
                                + 3*ddQ[q][3][1][1], 2)
                    ) * fe_values.JxW(q);

//                (
//                 (2*alpha*(-Q_vec[q][0]*Q_vec[q][0] - Q_vec[q][0]*Q_vec[q][3] 
//                           - Q_vec[q][1]*Q_vec[q][1] - Q_vec[q][2]*Q_vec[q][2] 
//                           - Q_vec[q][3]*Q_vec[q][3] - Q_vec[q][4]*Q_vec[q][4]))
//                 +
//                  (2*Q_vec[q][0]*Lambda_vec[0] + Q_vec[q][0]*Lambda_vec[3] 
//                   + 2*Q_vec[q][1]*Lambda_vec[1] + 2*Q_vec[q][2]*Lambda_vec[2] 
//                   + Q_vec[q][3]*Lambda_vec[0] + 2*Q_vec[q][3]*Lambda_vec[3] 
//                   + 2*Q_vec[q][4]*Lambda_vec[4] 
//                   + std::log(4*M_PI)
//                   - std::log(Z))
//                 +
//                  ((1.0/2.0)*dQ[q][0][0]*dQ[q][0][0] + dQ[q][0][1]*dQ[q][1][0] 
//                   + (1.0/2.0)*dQ[q][1][0]*dQ[q][1][0] + (1.0/2.0)*dQ[q][1][1]*dQ[q][1][1] 
//                   + dQ[q][1][1]*dQ[q][3][0] + (1.0/2.0)*dQ[q][2][0]*dQ[q][2][0] 
//                   + dQ[q][2][1]*dQ[q][4][0] + (1.0/2.0)*dQ[q][3][1]*dQ[q][3][1] 
//                   + (1.0/2.0)*dQ[q][4][1]*dQ[q][4][1])
//                +
//                  ((1.0/2.0)*L2*(dQ[q][0][0] + dQ[q][1][1]*dQ[q][0][0] 
//                      + dQ[q][1][1] + dQ[q][1][0] + dQ[q][3][1]*dQ[q][1][0] 
//                      + dQ[q][3][1] + dQ[q][2][0] + dQ[q][4][1]*dQ[q][2][0] 
//                      + dQ[q][4][1]))
//                +
//                  ((1.0/2.0)*L3*(2*((-dQ[q][0][0] - dQ[q][3][0])*(-dQ[q][0][1] - dQ[q][3][1]) 
//                          + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] 
//                          + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] 
//                          + 2*dQ[q][4][0]*dQ[q][4][1])*Q_vec[q][1] 
//                      + (-dQ[q][0][0] - dQ[q][3][0]*-dQ[q][0][0] - dQ[q][3][0] 
//                          + dQ[q][0][0]*dQ[q][0][0] + 2*dQ[q][1][0]*dQ[q][1][0] 
//                          + 2*dQ[q][2][0]*dQ[q][2][0] + dQ[q][3][0]*dQ[q][3][0] 
//                          + 2*dQ[q][4][0]*dQ[q][4][0])*Q_vec[q][0] 
//                      + (-dQ[q][0][1] - dQ[q][3][1]*-dQ[q][0][1] - dQ[q][3][1] 
//                          + dQ[q][0][1]*dQ[q][0][1] + 2*dQ[q][1][1]*dQ[q][1][1] 
//                          + 2*dQ[q][2][1]*dQ[q][2][1] + dQ[q][3][1]*dQ[q][3][1] 
//                          + 2*dQ[q][4][1]*dQ[q][4][1])*Q_vec[q][3]))
//                  )
//                *
//                (
//                 (2*alpha*(-Q_vec[q][0]*Q_vec[q][0] - Q_vec[q][0]*Q_vec[q][3] 
//                           - Q_vec[q][1]*Q_vec[q][1] - Q_vec[q][2]*Q_vec[q][2] 
//                           - Q_vec[q][3]*Q_vec[q][3] - Q_vec[q][4]*Q_vec[q][4]))
//                 +
//                  (2*Q_vec[q][0]*Lambda_vec[0] + Q_vec[q][0]*Lambda_vec[3] 
//                   + 2*Q_vec[q][1]*Lambda_vec[1] + 2*Q_vec[q][2]*Lambda_vec[2] 
//                   + Q_vec[q][3]*Lambda_vec[0] + 2*Q_vec[q][3]*Lambda_vec[3] 
//                   + 2*Q_vec[q][4]*Lambda_vec[4] 
//                   + std::log(4*M_PI)
//                   - std::log(Z))
//                 +
//                  ((1.0/2.0)*dQ[q][0][0]*dQ[q][0][0] + dQ[q][0][1]*dQ[q][1][0] 
//                   + (1.0/2.0)*dQ[q][1][0]*dQ[q][1][0] + (1.0/2.0)*dQ[q][1][1]*dQ[q][1][1] 
//                   + dQ[q][1][1]*dQ[q][3][0] + (1.0/2.0)*dQ[q][2][0]*dQ[q][2][0] 
//                   + dQ[q][2][1]*dQ[q][4][0] + (1.0/2.0)*dQ[q][3][1]*dQ[q][3][1] 
//                   + (1.0/2.0)*dQ[q][4][1]*dQ[q][4][1])
//                +
//                  ((1.0/2.0)*L2*(dQ[q][0][0] + dQ[q][1][1]*dQ[q][0][0] 
//                      + dQ[q][1][1] + dQ[q][1][0] + dQ[q][3][1]*dQ[q][1][0] 
//                      + dQ[q][3][1] + dQ[q][2][0] + dQ[q][4][1]*dQ[q][2][0] 
//                      + dQ[q][4][1]))
//                +
//                  ((1.0/2.0)*L3*(2*((-dQ[q][0][0] - dQ[q][3][0])*(-dQ[q][0][1] - dQ[q][3][1]) 
//                          + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] 
//                          + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] 
//                          + 2*dQ[q][4][0]*dQ[q][4][1])*Q_vec[q][1] 
//                      + (-dQ[q][0][0] - dQ[q][3][0]*-dQ[q][0][0] - dQ[q][3][0] 
//                          + dQ[q][0][0]*dQ[q][0][0] + 2*dQ[q][1][0]*dQ[q][1][0] 
//                          + 2*dQ[q][2][0]*dQ[q][2][0] + dQ[q][3][0]*dQ[q][3][0] 
//                          + 2*dQ[q][4][0]*dQ[q][4][0])*Q_vec[q][0] 
//                      + (-dQ[q][0][1] - dQ[q][3][1]*-dQ[q][0][1] - dQ[q][3][1] 
//                          + dQ[q][0][1]*dQ[q][0][1] + 2*dQ[q][1][1]*dQ[q][1][1] 
//                          + 2*dQ[q][2][1]*dQ[q][2][1] + dQ[q][3][1]*dQ[q][3][1] 
//                          + 2*dQ[q][4][1]*dQ[q][4][1])*Q_vec[q][3]))
//                  )
//                  * fe_values.JxW(q);
        }
    }

    double total_mean_field_term
        = dealii::Utilities::MPI::sum(mean_field_term, mpi_communicator);
    double total_entropy_term
        = dealii::Utilities::MPI::sum(entropy_term, mpi_communicator);
    double total_L1_elastic_term
        = dealii::Utilities::MPI::sum(L1_elastic_term, mpi_communicator);
    double total_L2_elastic_term
        = dealii::Utilities::MPI::sum(L2_elastic_term, mpi_communicator);
    double total_L3_elastic_term
        = dealii::Utilities::MPI::sum(L3_elastic_term, mpi_communicator);
    double total_dE_dQ_squared
        = dealii::Utilities::MPI::sum(dE_dQ_squared, mpi_communicator);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        energy_vals[0].push_back(current_time);
        energy_vals[1].push_back(total_mean_field_term);
        energy_vals[2].push_back(total_entropy_term);
        energy_vals[3].push_back(total_L1_elastic_term);
        energy_vals[4].push_back(total_L2_elastic_term);
        energy_vals[5].push_back(total_L3_elastic_term);
        energy_vals[6].push_back(dE_dQ_squared);
    }
}



template <int dim>
void NematicSystemMPI<dim>::
output_defect_positions(const MPI_Comm &mpi_communicator,
                        const std::string data_folder,
                        const std::string filename)
{
    std::vector<std::string> datanames = {"t", "x", "y"};
    if (dim == 3)
        datanames.push_back("z");
    datanames.push_back("charge");

    Output::distributed_vector_to_hdf5(defect_pts, 
                                       datanames, 
                                       mpi_communicator, 
                                       data_folder + filename 
                                       + std::string(".h5"));
}



template <int dim>
void NematicSystemMPI<dim>::
output_configuration_energies(const MPI_Comm &mpi_communicator,
                              const std::string data_folder,
                              const std::string filename)
{
    std::vector<std::string> datanames = {"t", 
                                          "mean_field_term",
                                          "entropy_term",
                                          "L1_elastic_term",
                                          "L2_elastic_term",
                                          "L3_elastic_term",
                                          "dE_dQ_squared"};

    Output::distributed_vector_to_hdf5(energy_vals, 
                                       datanames, 
                                       mpi_communicator, 
                                       data_folder + filename 
                                       + std::string(".h5"));
}



template <int dim>
void NematicSystemMPI<dim>::
output_results(const MPI_Comm &mpi_communicator,
               const dealii::parallel::distributed::Triangulation<dim>
               &triangulation,
               const std::string folder,
               const std::string filename,
               const int time_step) const
{
    NematicPostprocessor<dim> nematic_postprocessor;
    EnergyPostprocessor<dim> energy_postprocessor(lagrange_multiplier, 
                                                  maier_saupe_alpha, 
                                                  L2, 
                                                  L3);
    ConfigurationForcePostprocessor<dim> 
        configuration_force_postprocessor(lagrange_multiplier, 
                                          maier_saupe_alpha, 
                                          L2, 
                                          L3);
    dealii::DataOut<dim> data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, nematic_postprocessor);
    data_out.add_data_vector(current_solution, energy_postprocessor);
    data_out.add_data_vector(current_solution, configuration_force_postprocessor);
    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    std::ofstream output(folder + filename
                         + "_" + std::to_string(time_step)
                         + ".vtu");
    data_out.write_vtu_with_pvtu_record(folder, filename, time_step,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



template <int dim>
void NematicSystemMPI<dim>::
output_Q_components(const MPI_Comm &mpi_communicator,
                    const dealii::parallel::distributed::Triangulation<dim>
                    &triangulation,
                    const std::string folder,
                    const std::string filename,
                    const int time_step) const
{
    dealii::DataOut<dim> data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> Q_names(msc::vec_dim<dim>);
    for (std::size_t i = 0; i < Q_names.size(); ++i)
        Q_names[i] = std::string("Q") + std::to_string(i);

    data_out.add_data_vector(current_solution, Q_names);
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



template <int dim>
void NematicSystemMPI<dim>::
output_rhs_components(const MPI_Comm &mpi_communicator,
                      const dealii::parallel::distributed::Triangulation<dim>
                      &triangulation,
                      const std::string folder,
                      const std::string filename,
                      const int time_step) const
{
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> rhs_names(msc::vec_dim<dim>);
    for (std::size_t i = 0; i < rhs_names.size(); ++i)
        rhs_names[i] = std::string("rhs_") + std::to_string(i);

    data_out.add_data_vector(system_rhs, rhs_names);
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




template <int dim>
const dealii::DoFHandler<dim> &
NematicSystemMPI<dim>::return_dof_handler() const
{
    return dof_handler;
}



template <int dim>
const LA::MPI::Vector &
NematicSystemMPI<dim>::return_current_solution() const
{
    return current_solution;
}



template <int dim>
const dealii::AffineConstraints<double>&
NematicSystemMPI<dim>::return_constraints() const
{
    return constraints;
}


template <int dim>
double NematicSystemMPI<dim>::return_parameters() const
{
    return maier_saupe_alpha;
}


template <int dim>
void NematicSystemMPI<dim>::
set_current_solution(const MPI_Comm &mpi_communicator,
                     const LA::MPI::Vector &distributed_solution)
{
    current_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    current_solution = distributed_solution;
}

template class NematicSystemMPI<2>;
template class NematicSystemMPI<3>;
