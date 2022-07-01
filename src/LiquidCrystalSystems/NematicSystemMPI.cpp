#include "NematicSystemMPI.hpp"

#include <deal.II/distributed/tria.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
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
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

// #include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <boost/program_options.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/any.hpp>

#include "Utilities/maier_saupe_constants.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Postprocessors/DirectorPostprocessor.hpp"
#include "Postprocessors/SValuePostprocessor.hpp"
#include "Postprocessors/EvaluateFEObject.hpp"
#include "Postprocessors/NematicPostprocessor.hpp"

#include <deal.II/numerics/vector_tools_boundary.h>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>

namespace msc = maier_saupe_constants;



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
{}



template <int dim>
void NematicSystemMPI<dim>::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Nematic system MPI");

    prm.enter_subsection("Boundary values");
    prm.declare_entry("Name",
                      "uniform",
                      dealii::Patterns::Selection("uniform|periodic|defect|two-defect"));
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
        dealii::VectorTools::
            interpolate_boundary_values(dof_handler,
                                        /* boundary_component = */0,
                                        dealii::Functions::ZeroFunction<dim>(),
                                        constraints);
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
                      // locally_relevant_dofs,
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
    current_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    past_solution.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);

    // interpolate initial condition
    LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                           mpi_communicator);
    dealii::VectorTools::interpolate(dof_handler,
                                     *boundary_value_func,
                                     locally_owned_solution);
    locally_owned_solution.compress(dealii::VectorOperation::insert);
    current_solution = locally_owned_solution;
    past_solution = locally_owned_solution;

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

    configuration_constraints.distribute(current_solution);
    configuration_constraints.distribute(past_solution);
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
void NematicSystemMPI<dim>::solve_and_update(const MPI_Comm &mpi_communicator,
                                             const double alpha)
{
    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-10);
    // LA::SolverGMRES solver(solver_control, mpi_communicator);
    LA::SolverGMRES solver(solver_control);
    // dealii::PETScWrappers::PreconditionNone preconditioner;
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
const double NematicSystemMPI<dim>::return_parameters() const
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
