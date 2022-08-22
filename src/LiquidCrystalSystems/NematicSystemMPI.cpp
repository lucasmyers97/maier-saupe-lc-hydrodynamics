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
#include "Numerics/FindDefects.hpp"

#include <deal.II/numerics/vector_tools_boundary.h>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <utility>

namespace
{
    namespace msc = maier_saupe_constants;
}



template <int dim>
NematicSystemMPI<dim>::
NematicSystemMPI(const dealii::parallel::distributed::Triangulation<dim>
                 &triangulation,
                 const unsigned int degree,
                 const std::string boundary_values_name,
                 const std::map<std::string, boost::any> &am,
                 const double maier_saupe_alpha_,
                 const int order,
                 const double lagrange_step_size,
                 const double lagrange_tol,
                 const unsigned int lagrange_max_iters)
    : dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(degree), msc::vec_dim<dim>)
    , boundary_value_func(BoundaryValuesFactory::
                          BoundaryValuesFactory<dim>(boundary_values_name, am))
    , lagrange_multiplier(order,
                          lagrange_step_size,
                          lagrange_tol,
                          lagrange_max_iters)

    , maier_saupe_alpha(maier_saupe_alpha_)

    , defect_pts(dim + 1)
{}



template <int dim>
void NematicSystemMPI<dim>::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Nematic system MPI");

    prm.enter_subsection("Boundary values");
    prm.declare_entry("Name",
                      "uniform",
                      dealii::Patterns::Selection("uniform|periodic"
                                                  "|defect|two-defect"));
    prm.declare_entry("S value",
                      "0.6751",
                      dealii::Patterns::Double());
    prm.declare_entry("Phi",
                      "0.0",
                      dealii::Patterns::Double());
    prm.declare_entry("K",
                      "1.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Eps",
                      "0.1",
                      dealii::Patterns::Double());
    prm.declare_entry("Defect charge name",
                      "plus-half",
                      dealii::Patterns::Selection("plus-half|minus-half"
                                                  "|plus-one|minus-one"
                                                  "|plus-half-minus-half"
                                                  "|plus-half-minus-half-alt"));
    prm.declare_entry("Center x1",
                      "5.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Center y1",
                      "0.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Center x2",
                      "-5.0",
                      dealii::Patterns::Double());
    prm.declare_entry("Center y2",
                      "0.0",
                      dealii::Patterns::Double());
    prm.leave_subsection();

    prm.declare_entry("Maier saupe alpha",
                      "8.0",
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

    prm.enter_subsection("Boundary values");
    std::string boundary_values_name = prm.get("Name");
    std::map<std::string, boost::any> am;
    am["S-value"] = prm.get_double("S value");
    am["phi"] = prm.get_double("Phi");
    am["k"] = prm.get_double("K");
    am["eps"] = prm.get_double("Eps");
    am["defect-charge-name"] = prm.get("Defect charge name");
    double x1 = prm.get_double("Center x1");
    double y1 = prm.get_double("Center y1");
    double x2 = prm.get_double("Center x2");
    double y2 = prm.get_double("Center y2");
    am["centers"] = std::vector<double>({x1, y1, x2, y2});
    boundary_value_func = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(boundary_values_name, am);
    prm.leave_subsection();

    maier_saupe_alpha = prm.get_double("Maier saupe alpha");
    int order = prm.get_integer("Lebedev order");
    double lagrange_step_size = prm.get_double("Lagrange step size");
    double lagrange_tol = prm.get_double("Lagrange tolerance");
    int lagrange_max_iter = prm.get_integer("Lagrange maximum iterations");

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
        // dealii::VectorTools::
        //     interpolate_boundary_values(dof_handler,
        //                                 /* boundary_component = */0,
        //                                 dealii::Functions::ZeroFunction<dim>(msc::vec_dim<dim>),
        //                                 constraints);
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
assemble_system_anisotropic(double dt, const MPI_Comm &mpi_communicator)
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
    const double L2 = 0;
    const double L3 = 3.0;

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    // auto-diff stuff -----------------
    unsigned int n_locally_active_cells 
        = dof_handler.get_triangulation().n_active_cells();
    std::vector<double> matrix_cell_residual1(n_locally_active_cells);
    std::vector<double> cell_x(n_locally_active_cells);
    std::vector<double> cell_y(n_locally_active_cells);
    dealii::Point<dim> cell_pt;

    using ADHelper 
        = dealii::Differentiation::AD::ResidualLinearization<
            dealii::Differentiation::AD::NumberTypes::sacado_dfad,
            double>;
    using ADNumberType = typename ADHelper::ad_type;

    std::vector<dealii::FEValuesExtractors::Scalar> Q_fe(fe.components);
    for (std::size_t i = 0; i < Q_fe.size(); ++i)
        Q_fe[i] = dealii::FEValuesExtractors::Scalar(i);

    dealii::FullMatrix<double> ad1_cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> ad1_cell_rhs(dofs_per_cell);
    dealii::FullMatrix<double> an1_cell_matrix(dofs_per_cell, dofs_per_cell);

    const unsigned int n_independent_variables = local_dof_indices.size();
    const unsigned int n_dependent_variables   = dofs_per_cell;
    ADHelper ad1_helper(n_independent_variables, n_dependent_variables);

    std::vector<ADNumberType> ad1_Q_vec_temp(n_q_points);
    std::vector<std::vector<ADNumberType>>
        ad1_Q_vec(n_q_points, std::vector<ADNumberType>(fe.components));
    std::vector<dealii::Tensor<1, dim, ADNumberType>> ad1_dQ_temp(n_q_points);
    std::vector<std::vector<dealii::Tensor<1, dim, ADNumberType>>>
        ad1_dQ(n_q_points,
               std::vector<dealii::Tensor<1, dim, ADNumberType>>(fe.components));

    unsigned int cell_num = 0;

    // --
    
    std::vector<double> matrix_cell_residual2(n_locally_active_cells);

    dealii::FullMatrix<double> ad2_cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> ad2_cell_rhs(dofs_per_cell);
    dealii::FullMatrix<double> an2_cell_matrix(dofs_per_cell, dofs_per_cell);

    ADHelper ad2_helper(n_independent_variables, n_dependent_variables);

    std::vector<ADNumberType> ad2_Q_vec_temp(n_q_points);
    std::vector<std::vector<ADNumberType>>
        ad2_Q_vec(n_q_points, std::vector<ADNumberType>(fe.components));
    std::vector<dealii::Tensor<1, dim, ADNumberType>> ad2_dQ_temp(n_q_points);
    std::vector<std::vector<dealii::Tensor<1, dim, ADNumberType>>>
        ad2_dQ(n_q_points,
               std::vector<dealii::Tensor<1, dim, ADNumberType>>(fe.components));
    // ---------------------------

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if ( !(cell->is_locally_owned()) )
            continue;

        cell_matrix = 0;
        an1_cell_matrix = 0;
        an2_cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);

        fe_values.reinit(cell);
        fe_values.get_function_gradients(current_solution, dQ);
        fe_values.get_function_values(current_solution, Q_vec);
        fe_values.get_function_values(past_solution, Q0_vec);

        // auto-diff stuff ------------------
        ad1_helper.reset(n_independent_variables, n_dependent_variables);
        ad1_helper.register_dof_values(current_solution, local_dof_indices);
        const std::vector<ADNumberType> &dof_values_ad1 
            = ad1_helper.get_sensitive_dof_values();
        std::vector<ADNumberType> residual_ad1(n_dependent_variables,
                                              ADNumberType(0.0));

        for (std::size_t k = 0; k < Q_fe.size(); ++k)
        {
            fe_values[Q_fe[k]].
                get_function_values_from_local_dof_values(dof_values_ad1, 
                                                          ad1_Q_vec_temp);
            fe_values[Q_fe[k]].
                get_function_gradients_from_local_dof_values(dof_values_ad1, 
                                                             ad1_dQ_temp);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                ad1_Q_vec[q][k] = ad1_Q_vec_temp[q];
                ad1_dQ[q][k] = ad1_dQ_temp[q];
            }
        }

        //--
        ad2_helper.reset(n_independent_variables, n_dependent_variables);
        ad2_helper.register_dof_values(current_solution, local_dof_indices);
        const std::vector<ADNumberType> &dof_values_ad2 
            = ad2_helper.get_sensitive_dof_values();
        std::vector<ADNumberType> residual_ad2(n_dependent_variables,
                                              ADNumberType(0.0));

        for (std::size_t k = 0; k < Q_fe.size(); ++k)
        {
            fe_values[Q_fe[k]].
                get_function_values_from_local_dof_values(dof_values_ad2, 
                                                          ad2_Q_vec_temp);
            fe_values[Q_fe[k]].
                get_function_gradients_from_local_dof_values(dof_values_ad2, 
                                                             ad2_dQ_temp);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                ad2_Q_vec[q][k] = ad2_Q_vec_temp[q];
                ad2_dQ[q][k] = ad2_dQ_temp[q];
            }
        }
        // ---------------------------

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
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (dt*(2*dLambda_dQ[0][1] + dLambda_dQ[3][1])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (dt*(2*dLambda_dQ[0][2] + dLambda_dQ[3][2])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
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
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (dt*(2*dLambda_dQ[0][4] + dLambda_dQ[3][4])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[1][0]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
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
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[1][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[1][3]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[1][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[2][0]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[2][1]
                                  * fe_values.shape_value(i, q)
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
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[2][3]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[2][4]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
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
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (dt*(dLambda_dQ[0][1] + 2*dLambda_dQ[3][1])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (dt*(dLambda_dQ[0][2] + 2*dLambda_dQ[3][2])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
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
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 4)
                        cell_matrix(i, j) +=
                                (
                                 (dt*(dLambda_dQ[0][4] + 2*dLambda_dQ[3][4])
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 0)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[4][0]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 1)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[4][1]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 2)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[4][2]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 3)
                        cell_matrix(i, j) +=
                                (
                                 (2*dt*dLambda_dQ[4][3]
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q))
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
                                )
                                * fe_values.JxW(q);
                    
                    // auto-diff checking stuff
                    if (component_i == 0 && component_j == 0)
                        an1_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(2*(Q_vec[q][1]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] + ((dQ[q][0][0] + dQ[q][3][0])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][0] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][0] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][0][0]) * fe_values.shape_grad(i, q)[0]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 1)
                        an1_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(2*dQ[q][0][0]*fe_values.shape_grad(i, q)[1] 
                                  + 2*dQ[q][0][1] * fe_values.shape_grad(i, q)[0] 
                                  + dQ[q][3][0] * fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][3][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 2)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 3)
                        an1_cell_matrix(i, j) +=
                                (
                                 (L3*dt*((Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] + ((dQ[q][0][1] + dQ[q][3][1])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][0][1] * fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 4)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 0)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][1][0]*fe_values.shape_grad(i, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 1)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*((Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][1][1]) * fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][1][0]) * fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 2)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 3)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][1][1]*fe_values.shape_grad(i, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 4)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 0)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][2][0]*fe_values.shape_grad(i, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 1)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*(dQ[q][2][0]*fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][2][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 2)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*((Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 3)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][2][1]*fe_values.shape_grad(i, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 4)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 0)
                        an1_cell_matrix(i, j) +=
                                (
                                 (L3*dt*((Q_vec[q][1]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] + ((dQ[q][0][0] + dQ[q][3][0])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][0] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + fe_values.shape_value(j, q)*dQ[q][3][0] * fe_values.shape_grad(i, q)[0]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 1)
                        an1_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(dQ[q][0][0]*fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][0][1] * fe_values.shape_grad(i, q)[0] 
                                  + 2*dQ[q][3][0] * fe_values.shape_grad(i, q)[1] 
                                  + 2*dQ[q][3][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 2)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 3)
                        an1_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(2*(Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] + ((dQ[q][0][1] + dQ[q][3][1])
                                  * fe_values.shape_value(j, q) 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1] 
                                  + fe_values.shape_value(j, q)*dQ[q][3][1]) * fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 4)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 0)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][4][0]*fe_values.shape_grad(i, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 1)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*(dQ[q][4][0]*fe_values.shape_grad(i, q)[1] 
                                  + dQ[q][4][1] * fe_values.shape_grad(i, q)[0])
                                  * fe_values.shape_value(j, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 2)
                        an1_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 3)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(j, q)*dQ[q][4][1]*fe_values.shape_grad(i, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 4)
                        an1_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*((Q_vec[q][0]*fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][1] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[0] 
                                  + (Q_vec[q][1] * fe_values.shape_grad(j, q)[0] 
                                  + Q_vec[q][3] * fe_values.shape_grad(j, q)[1])*fe_values.shape_grad(i, q)[1]))
                                )
                                * fe_values.JxW(q);
                    // --
                    if (component_i == 0 && component_j == 0)
                        an2_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(2*dQ[q][0][0] + dQ[q][3][0])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 1)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][1][0]*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 2)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][2][0]*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 3)
                        an2_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(dQ[q][0][0] + 2*dQ[q][3][0])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 0 && component_j == 4)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][4][0]*fe_values.shape_grad(j, q)[0])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 0)
                        an2_cell_matrix(i, j) +=
                                (
                                 (L3*dt*((2*dQ[q][0][0] 
                                  + dQ[q][3][0]) * fe_values.shape_grad(j, q)[1] 
                                  + (2*dQ[q][0][1] + dQ[q][3][1]) * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 1)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*(dQ[q][1][0]*fe_values.shape_grad(j, q)[1] 
                                  + dQ[q][1][1] * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 2)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*(dQ[q][2][0]*fe_values.shape_grad(j, q)[1] 
                                  + dQ[q][2][1] * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 3)
                        an2_cell_matrix(i, j) +=
                                (
                                 (L3*dt*((dQ[q][0][0] 
                                  + 2*dQ[q][3][0]) * fe_values.shape_grad(j, q)[1] 
                                  + (dQ[q][0][1] + 2*dQ[q][3][1]) * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 1 && component_j == 4)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt*(dQ[q][4][0]*fe_values.shape_grad(j, q)[1] 
                                  + dQ[q][4][1] * fe_values.shape_grad(j, q)[0])
                                  * fe_values.shape_value(i, q))
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 0)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 1)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 2)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 3)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 2 && component_j == 4)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 0)
                        an2_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(2*dQ[q][0][1] + dQ[q][3][1])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 1)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][1][1]*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 2)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][2][1]*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 3)
                        an2_cell_matrix(i, j) +=
                                (
                                 (L3*dt*(dQ[q][0][1] + 2*dQ[q][3][1])
                                  * fe_values.shape_value(i, q)*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 3 && component_j == 4)
                        an2_cell_matrix(i, j) +=
                                (
                                 (2*L3*dt
                                  * fe_values.shape_value(i, q)*dQ[q][4][1]*fe_values.shape_grad(j, q)[1])
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 0)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 1)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 2)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 3)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);
                    else if (component_i == 4 && component_j == 4)
                        an2_cell_matrix(i, j) +=
                                (
                                 0
                                )
                                * fe_values.JxW(q);                    
                // -----------------------------
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
                         (dt*(
                          - 2*dQ[q][0][0] * fe_values.shape_grad(i, q)[0] 
                          - 2*dQ[q][0][1] * fe_values.shape_grad(i, q)[1] 
                          - dQ[q][3][0] * fe_values.shape_grad(i, q)[0] 
                          - dQ[q][3][1] * fe_values.shape_grad(i, q)[1]))
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
                         (dt*(
                          - 2*dQ[q][1][0] * fe_values.shape_grad(i, q)[0] 
                          - 2*dQ[q][1][1] * fe_values.shape_grad(i, q)[1]))
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
                         (dt*(
                          - 2*dQ[q][2][0] * fe_values.shape_grad(i, q)[0] 
                          - 2*dQ[q][2][1] * fe_values.shape_grad(i, q)[1]))
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
                         (dt*(
                          - dQ[q][0][0] * fe_values.shape_grad(i, q)[0] 
                          - dQ[q][0][1] * fe_values.shape_grad(i, q)[1] 
                          - 2*dQ[q][3][0] * fe_values.shape_grad(i, q)[0] 
                          - 2*dQ[q][3][1] * fe_values.shape_grad(i, q)[1]))
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
                         (dt*(
                          - 2*dQ[q][4][0] * fe_values.shape_grad(i, q)[0] 
                          - 2*dQ[q][4][1] * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);

                // autodiff stuf ------------------------
                if (component_i == 0)
                    residual_ad1[i] +=
                        (
                         (L3*dt*(((ad1_dQ[q][0][0] 
                          + ad1_dQ[q][3][0])*ad1_Q_vec[q][0] + (ad1_dQ[q][0][1] + ad1_dQ[q][3][1])*ad1_Q_vec[q][1]) * fe_values.shape_grad(i, q)[0] 
                          + ((ad1_dQ[q][0][0] + ad1_dQ[q][3][0])*ad1_Q_vec[q][1] + (ad1_dQ[q][0][1] + ad1_dQ[q][3][1])*ad1_Q_vec[q][3]) * fe_values.shape_grad(i, q)[1] 
                          + (ad1_Q_vec[q][0]*ad1_dQ[q][0][0] + ad1_Q_vec[q][1]*ad1_dQ[q][0][1]) * fe_values.shape_grad(i, q)[0] 
                          + (ad1_Q_vec[q][1]*ad1_dQ[q][0][0] + ad1_Q_vec[q][3]*ad1_dQ[q][0][1]) * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 1)
                    residual_ad1[i] +=
                        (
                         (2*L3*dt*((ad1_Q_vec[q][0]*ad1_dQ[q][1][0] 
                          + ad1_Q_vec[q][1]*ad1_dQ[q][1][1]) * fe_values.shape_grad(i, q)[0] 
                          + (ad1_Q_vec[q][1]*ad1_dQ[q][1][0] + ad1_Q_vec[q][3]*ad1_dQ[q][1][1]) * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 2)
                    residual_ad1[i] +=
                        (
                         (2*L3*dt*((ad1_Q_vec[q][0]*ad1_dQ[q][2][0] 
                          + ad1_Q_vec[q][1]*ad1_dQ[q][2][1]) * fe_values.shape_grad(i, q)[0] 
                          + (ad1_Q_vec[q][1]*ad1_dQ[q][2][0] + ad1_Q_vec[q][3]*ad1_dQ[q][2][1]) * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 3)
                    residual_ad1[i] +=
                        (
                         (L3*dt*(((ad1_dQ[q][0][0] 
                          + ad1_dQ[q][3][0])*ad1_Q_vec[q][0] + (ad1_dQ[q][0][1] + ad1_dQ[q][3][1])*ad1_Q_vec[q][1]) * fe_values.shape_grad(i, q)[0] 
                          + ((ad1_dQ[q][0][0] + ad1_dQ[q][3][0])*ad1_Q_vec[q][1] + (ad1_dQ[q][0][1] + ad1_dQ[q][3][1])*ad1_Q_vec[q][3]) * fe_values.shape_grad(i, q)[1] 
                          + (ad1_Q_vec[q][0]*ad1_dQ[q][3][0] + ad1_Q_vec[q][1]*ad1_dQ[q][3][1]) * fe_values.shape_grad(i, q)[0] 
                          + (ad1_Q_vec[q][1]*ad1_dQ[q][3][0] + ad1_Q_vec[q][3]*ad1_dQ[q][3][1]) * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 4)
                    residual_ad1[i] +=
                        (
                         (2*L3*dt*((ad1_Q_vec[q][0]*ad1_dQ[q][4][0] 
                          + ad1_Q_vec[q][1]*ad1_dQ[q][4][1]) * fe_values.shape_grad(i, q)[0] 
                          + (ad1_Q_vec[q][1]*ad1_dQ[q][4][0] + ad1_Q_vec[q][3]*ad1_dQ[q][4][1]) * fe_values.shape_grad(i, q)[1]))
                        )
                        * fe_values.JxW(q);
                // --
                if (component_i == 0)
                    residual_ad2[i] +=
                        (
                         (L3*dt*(ad2_dQ[q][0][0]*ad2_dQ[q][0][0] + ad2_dQ[q][0][0]*ad2_dQ[q][3][0] + ad2_dQ[q][1][0]*ad2_dQ[q][1][0] + ad2_dQ[q][2][0]*ad2_dQ[q][2][0] + ad2_dQ[q][3][0]*ad2_dQ[q][3][0] + ad2_dQ[q][4][0]*ad2_dQ[q][4][0])
                          * fe_values.shape_value(i, q))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 1)
                    residual_ad2[i] +=
                        (
                         (L3*dt*((ad2_dQ[q][0][0] + ad2_dQ[q][3][0])*(ad2_dQ[q][0][1] + ad2_dQ[q][3][1]) + ad2_dQ[q][0][0]*ad2_dQ[q][0][1] + 2*ad2_dQ[q][1][0]*ad2_dQ[q][1][1] + 2*ad2_dQ[q][2][0]*ad2_dQ[q][2][1] + ad2_dQ[q][3][0]*ad2_dQ[q][3][1] + 2*ad2_dQ[q][4][0]*ad2_dQ[q][4][1])
                          * fe_values.shape_value(i, q))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 2)
                    residual_ad2[i] +=
                        (
                         0
                        )
                        * fe_values.JxW(q);
                else if (component_i == 3)
                    residual_ad2[i] +=
                        (
                         (L3*dt*(ad2_dQ[q][0][1]*ad2_dQ[q][0][1] + ad2_dQ[q][0][1]*ad2_dQ[q][3][1] + ad2_dQ[q][1][1]*ad2_dQ[q][1][1] + ad2_dQ[q][2][1]*ad2_dQ[q][2][1] + ad2_dQ[q][3][1]*ad2_dQ[q][3][1] + ad2_dQ[q][4][1]*ad2_dQ[q][4][1])
                          * fe_values.shape_value(i, q))
                        )
                        * fe_values.JxW(q);
                else if (component_i == 4)
                    residual_ad2[i] +=
                        (
                         0
                        )
                        * fe_values.JxW(q);
                // ------------------------
            }
        }
        // autodiff stuff -----------------
        ad1_helper.register_residual_vector(residual_ad1);
 
        ad1_helper.compute_residual(ad1_cell_rhs);
        ad1_cell_rhs *= -1.0;
 
        ad1_helper.compute_linearization(ad1_cell_matrix);

        cell_rhs += ad1_cell_rhs;
        cell_matrix.add(1.0, ad1_cell_matrix);

        an1_cell_matrix.add(-1.0, ad1_cell_matrix);
        matrix_cell_residual1[cell_num] = an1_cell_matrix.frobenius_norm();
        //--

        ad2_helper.register_residual_vector(residual_ad2);
 
        ad2_helper.compute_residual(ad2_cell_rhs);
        ad2_cell_rhs *= -1.0;
 
        ad2_helper.compute_linearization(ad2_cell_matrix);

        cell_rhs += ad2_cell_rhs;
        cell_matrix.add(1.0, ad2_cell_matrix);

        an2_cell_matrix.add(-1.0, ad2_cell_matrix);
        matrix_cell_residual2[cell_num] = an2_cell_matrix.frobenius_norm();
        // -------------------------------
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);

        cell_num++;
    }
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);

    // autodiff stuff ------------------------
    std::vector<std::vector<double>> res_data(4);
    res_data[0] = std::move(matrix_cell_residual1);
    res_data[1] = std::move(matrix_cell_residual2);
    res_data[2] = std::move(cell_x);
    res_data[3] = std::move(cell_y);

    std::vector<std::string> data_names = {"matrix_residual1", "matrix_residual2", "x", "y"};

    Output::distributed_vector_to_hdf5(res_data, 
                                       data_names, 
                                       mpi_communicator, 
                                       std::string("matrix_residual_data.h5"));      
    // --------------------------------------
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
             unsigned int current_timestep)
{
    auto local_minima = NumericalTools::find_defects(dof_handler, 
                                                     current_solution, 
                                                     min_dist, 
                                                     charge_threshold);
    for (const auto &pt : local_minima)
    {
        defect_pts[0].push_back(current_timestep);
        defect_pts[1].push_back(pt[0]);
        defect_pts[2].push_back(pt[1]);
        if (dim == 3)
            defect_pts[3].push_back(pt[2]);
    }
}



template <int dim>
void NematicSystemMPI<dim>::
output_defect_positions(const MPI_Comm &mpi_communicator,
                        const std::string data_folder,
                        const std::string filename)
{
    unsigned int this_process 
        = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    // vector with length of each set of defect points, indexed by process
    std::vector<std::size_t> process_data_lengths
        = dealii::Utilities::MPI::all_gather(mpi_communicator, 
                                             defect_pts[0].size());
    auto this_process_iter 
        = std::next(process_data_lengths.begin(), this_process);
    hsize_t write_index = std::accumulate(process_data_lengths.begin(), 
                                          this_process_iter, 
                                          0);
    hsize_t total_data_length = std::accumulate(process_data_lengths.begin(), 
                                                process_data_lengths.end(), 
                                                0);

    std::vector<hsize_t> dataset_dims = {total_data_length};
    std::vector<hsize_t> hyperslab_offset = {write_index};
    std::vector<hsize_t> hyperslab_dims = {process_data_lengths[this_process]};

    std::string group_name("defect");
    std::string t_name("t");
    std::string x_name("x");
    std::string y_name("y");

    dealii::HDF5::File file(data_folder + filename + std::string(".h5"), 
                            dealii::HDF5::File::FileAccessMode::create,
                            mpi_communicator);
    auto group = file.create_group(group_name);

    auto t_dataset = group.create_dataset<double>(t_name, dataset_dims);
    auto x_dataset = group.create_dataset<double>(x_name, dataset_dims);
    auto y_dataset = group.create_dataset<double>(y_name, dataset_dims);

    t_dataset.write_hyperslab(defect_pts[0], hyperslab_offset, hyperslab_dims);
    x_dataset.write_hyperslab(defect_pts[1], hyperslab_offset, hyperslab_dims);
    y_dataset.write_hyperslab(defect_pts[2], hyperslab_offset, hyperslab_dims);

    if (dim == 3)
    {
        std::string z_name("z");
        auto z_dataset = group.create_dataset<double>(z_name, dataset_dims);
        z_dataset.write_hyperslab(defect_pts[3], 
                                  hyperslab_offset, 
                                  hyperslab_dims);
    }
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
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, nematic_postprocessor);
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
