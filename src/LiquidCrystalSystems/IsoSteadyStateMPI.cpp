#include "IsoSteadyStateMPI.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/base/types.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/vector_operation.h>

namespace LA = dealii::LinearAlgebraPETSc;

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/base/tensor.h>
#include <deal.II/lac/lapack_full_matrix.h>

// May or may not need these
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/grid_out.h>
// ------

#include <boost/program_options.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <highfive/H5Easy.hpp>

#include "maier_saupe_constants.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "LagrangeMultiplier.hpp"
#include "Postprocessors/DirectorPostprocessor.hpp"
#include "Postprocessors/SValuePostprocessor.hpp"
#include "Postprocessors/EvaluateFEObject.hpp"
#include "Utilities/LinearInterpolation.hpp"

#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>

#include <deal.II/fe/mapping_q1.h>

namespace po = boost::program_options;
namespace msc = maier_saupe_constants;



template <int dim, int order>
IsoSteadyStateMPI<dim, order>::IsoSteadyStateMPI(const po::variables_map &vm)
    : mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening))
    , fe(dealii::FE_Q<dim>(1), msc::vec_dim<dim>)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      dealii::TimerOutput::summary,
                      dealii::TimerOutput::wall_times)

    , lagrange_multiplier(vm["lagrange-step-size"].as<double>(),
                          vm["lagrange-tol"].as<double>(),
                          vm["lagrange-max-iters"].as<int>())
    , boundary_value_func(BoundaryValuesFactory::BoundaryValuesFactory<dim>(vm))

    , left_endpoint(vm["left-endpoint"].as<double>())
    , right_endpoint(vm["right-endpoint"].as<double>())
    , num_refines(vm["num-refines"].as<int>())

    , simulation_step_size(vm["simulation-step-size"].as<double>())
    , simulation_tol(vm["simulation-tol"].as<double>())
    , simulation_max_iters(vm["simulation-max-iters"].as<int>())
    , maier_saupe_alpha(vm["maier-saupe-alpha"].as<double>())

    , boundary_values_name(vm["boundary-values-name"].as<std::string>())
    , S_value(vm["S-value"].as<double>())
    , defect_charge_name(vm["defect-charge-name"].as<std::string>())

    , data_folder(vm["data-folder"].as<std::string>())
    , initial_config_filename(vm["initial-config-filename"].as<std::string>())
    , final_config_filename(vm["final-config-filename"].as<std::string>())
    , archive_filename(vm["archive-filename"].as<std::string>())
{
    MPI_Comm_rank(mpi_communicator, &rank);
    MPI_Comm_size(mpi_communicator, &num_ranks);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::make_grid(const unsigned int num_refines,
                                              const double left,
                                              const double right)
{
    dealii::GridGenerator::hyper_cube(triangulation, left, right);
    triangulation.refine_global(num_refines);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::setup_system(bool initial_step)
{
    dealii::TimerOutput::Scope t(computing_timer, "setup");
    if (initial_step)
    {
        dof_handler.distribute_dofs(fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        locally_relevant_solution.reinit(locally_owned_dofs,
                                         locally_relevant_dofs,
                                         mpi_communicator);

        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler,
                                                         0,
                                                         *boundary_value_func,
                                                         constraints);
        constraints.close();

        LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                               mpi_communicator);
        dealii::AffineConstraints<double> locally_owned_constraints;
        locally_owned_constraints.clear();
        locally_owned_constraints.reinit(locally_owned_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, locally_owned_constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler,
                                                         0,
                                                         *boundary_value_func,
                                                         locally_owned_constraints);
        locally_owned_constraints.close();

        dealii::VectorTools::interpolate(dof_handler,
                                         *boundary_value_func,
                                         locally_owned_solution);

        locally_owned_solution.compress(dealii::VectorOperation::insert);
        locally_relevant_solution = locally_owned_solution;
        constraints.distribute(locally_relevant_solution);
        locally_relevant_solution.compress(dealii::VectorOperation::add);

        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler,
                                                         0,
                                                         dealii::Functions::ZeroFunction<dim>(msc::vec_dim<dim>),
                                                         constraints);
        constraints.close();
    }

    locally_relevant_update.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);
    system_rhs.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    rhs_term1.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);
    rhs_term2.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);
    rhs_term3.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);

    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                       dof_handler.locally_owned_dofs(),
                                                       mpi_communicator,
                                                       locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
    rhs_term1.compress(dealii::VectorOperation::add);
    rhs_term2.compress(dealii::VectorOperation::add);
    rhs_term3.compress(dealii::VectorOperation::add);
}




template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::assemble_system(int step)
{
    dealii::TimerOutput::Scope t(computing_timer, "assembly");

    const dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values
                                    | dealii::update_quadrature_points);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);
    dealii::Vector<double> cell_rhs_term1(dofs_per_cell);
    dealii::Vector<double> cell_rhs_term2(dofs_per_cell);
    dealii::Vector<double> cell_rhs_term3(dofs_per_cell);

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        old_solution_gradients
        (n_q_points,
         std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        old_solution_values(n_q_points, dealii::Vector<double>(fe.components));
    dealii::Vector<double> Lambda(fe.components);
    dealii::LAPACKFullMatrix<double> R(fe.components, fe.components);
    std::vector<dealii::Vector<double>>
        R_inv_phi(dofs_per_cell, dealii::Vector<double>(fe.components));

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    completely_distributed_solution.compress(dealii::VectorOperation::insert);
    completely_distributed_solution = locally_relevant_solution;
    completely_distributed_solution.compress(dealii::VectorOperation::add);

    // For debugging purposes --------------------------
    // dealii::MappingQ1<dim> mapping;
    // unsigned int n_active_cells = triangulation.n_active_cells();
    // std::vector<std::vector<double>> support_points(n_active_cells*dofs_per_cell, std::vector<double>(dim));
    // std::vector<double> support_point_vals(n_active_cells*dofs_per_cell);
    // std::vector<int> is_locally_owned(n_active_cells*dofs_per_cell, 0);

    // unsigned int total_quad_points = n_q_points * triangulation.n_active_cells();
    // std::vector<std::vector<double>> all_quad_points(total_quad_points, std::vector<double>(dim));
    // std::vector<std::vector<double>> all_Q_vals(total_quad_points, std::vector<double>(msc::vec_dim<dim>));
    // unsigned int cell_idx = 0;
    // -------------------------------------------------

    locally_relevant_solution.update_ghost_values();
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
          // For debugging purposes --------------------------
          // std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
          // cell->get_dof_indices(dof_indices);

          // const std::vector<dealii::Point<dim>> unit_points = fe.get_unit_support_points();
          // unsigned int n_support_points = unit_points.size();
          // for (unsigned int i = 0; i < n_support_points; i++)
          // {
          //   for (unsigned int j = 0; j < dim; j++)
          //     support_points[i + cell_idx * n_support_points][j] =
          //         mapping.transform_unit_to_real_cell(cell, unit_points[i])[j];

          //     support_point_vals[i + cell_idx * n_support_points] =
          //       locally_relevant_solution[dof_indices[i]];

          //     is_locally_owned[i + cell_idx * n_support_points] = 1;
          // }
          // -------------------------------------------------

            cell_matrix = 0.;
            cell_rhs = 0.;
            cell_rhs_term1 = 0.;
            cell_rhs_term1 = 0.;
            cell_rhs_term1 = 0.;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(locally_relevant_solution,
                                             old_solution_gradients);
            fe_values.get_function_values(locally_relevant_solution,
                                          old_solution_values);
            // fe_values.get_function_gradients(completely_distributed_solution,
            //                                  old_solution_gradients);
            // fe_values.get_function_values(completely_distributed_solution,
            //                               old_solution_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                // For debugging purposes ------------------------
                // dealii::Point<dim> quad_point = fe_values.quadrature_point(q);
                // for (int i = 0; i < dim; i++)
                //     all_quad_points[q + n_q_points * cell_idx][i] = quad_point[i];

                // for (int i = 0; i < msc::vec_dim<dim>; i++)
                //     all_Q_vals[q + n_q_points * cell_idx][i] = old_solution_values[q][i];
                // -----------------------------------------------

                Lambda.reinit(fe.components);
                R.reinit(fe.components);

                lagrange_multiplier.invertQ(old_solution_values[q]);
                lagrange_multiplier.returnLambda(Lambda);
                lagrange_multiplier.returnJac(R);
                R.compute_lu_factorization();

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const unsigned int component_j =
                        fe.system_to_component_index(j).first;
                    R_inv_phi[j].reinit(fe.components);
                    R_inv_phi[j][component_j] = fe_values.shape_value(j, q);
                    R.solve(R_inv_phi[j]);
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
                               * maier_saupe_alpha
                               * fe_values.shape_value(j, q)) :
                              0)
                             -
                             ((component_i == component_j) ?
                              (fe_values.shape_grad(i, q)
                               * fe_values.shape_grad(j, q)) :
                              0)
                             -
                             (fe_values.shape_value(i, q)
                              * R_inv_phi[j][component_i]))
                            * fe_values.JxW(q);
                    }
                    cell_rhs(i) +=
                        (-(fe_values.shape_value(i, q)
                           * maier_saupe_alpha
                           * old_solution_values[q][component_i])
                         +
                         (fe_values.shape_grad(i, q)
                          * old_solution_gradients[q][component_i])
                         +
                         (fe_values.shape_value(i, q)
                          * Lambda[component_i]))
                        * fe_values.JxW(q);

                    cell_rhs_term1(i) +=
                        (-(fe_values.shape_value(i, q)
                           * maier_saupe_alpha
                           * old_solution_values[q][component_i])
                         ) * fe_values.JxW(q);
                    cell_rhs_term2(i) +=
                         (fe_values.shape_grad(i, q)
                          * old_solution_gradients[q][component_i])
                        * fe_values.JxW(q);
                    cell_rhs_term3(i) +=
                         (fe_values.shape_value(i, q)
                          * Lambda[component_i])
                        * fe_values.JxW(q);

                } // i_loop
            } // quadrature points loop
            // std::cout << "\n\n";

            cell->get_dof_indices(local_dof_indices);
            // constraints.distribute_local_to_global(cell_matrix,
            //                                        cell_rhs,
            //                                        local_dof_indices,
            //                                        system_matrix,
            //                                        system_rhs);
            constraints.distribute_local_to_global(cell_matrix,
                                                   local_dof_indices,
                                                   system_matrix);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
            constraints.distribute_local_to_global(cell_rhs_term1,
                                                   local_dof_indices,
                                                   rhs_term1);
            constraints.distribute_local_to_global(cell_rhs_term2,
                                                   local_dof_indices,
                                                   rhs_term2);
            constraints.distribute_local_to_global(cell_rhs_term3,
                                                   local_dof_indices,
                                                   rhs_term3);
        } // if_locally_owned loop
        // cell_idx++;
    } // cell loop

    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
    rhs_term1.compress(dealii::VectorOperation::add);
    rhs_term2.compress(dealii::VectorOperation::add);
    rhs_term3.compress(dealii::VectorOperation::add);

    // std::string filename = "./assemble_system_output_" + std::to_string(step)
    //     + "_" + std::to_string(rank) + "_" + std::to_string(num_ranks) + ".h5";
    // H5Easy::File file(filename, H5Easy::File::Overwrite);
    // H5Easy::dump(file, "/points", all_quad_points);
    // H5Easy::dump(file, "/Q_vals", all_Q_vals);
    // H5Easy::dump(file, "/support_points", support_points);
    // H5Easy::dump(file, "/support_point_vals", support_point_vals);
    // H5Easy::dump(file, "/is_locally_owned", is_locally_owned);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::solve()
{
    dealii::TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

    LA::SolverGMRES solver(solver_control, mpi_communicator);

    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;

    data.symmetric_operator = false;

    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    constraints.distribute(completely_distributed_solution);

    const double newton_alpha = determine_step_length();
    completely_distributed_solution *= newton_alpha;
    LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                           mpi_communicator);
    locally_owned_solution = locally_relevant_solution;
    locally_owned_solution += completely_distributed_solution;
    locally_relevant_solution = locally_owned_solution;
    locally_relevant_solution.compress(dealii::VectorOperation::insert);
}



template <int dim, int order>
double IsoSteadyStateMPI<dim, order>::determine_step_length()
{
  return simulation_step_size;
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::output_results
(const std::string folder, const std::string filename, const int step) const
{
    DirectorPostprocessor<dim>
        director_postprocessor_defect(boundary_values_name);
    SValuePostprocessor<dim> S_value_postprocessor_defect(boundary_values_name);
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, director_postprocessor_defect);
    data_out.add_data_vector(locally_relevant_solution, S_value_postprocessor_defect);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(folder, filename, step,
                                        mpi_communicator, 2, 8);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::output_update
(const std::string folder, const std::string filename, const int step) const
{
    DirectorPostprocessor<dim>
        director_postprocessor_defect(boundary_values_name);
    SValuePostprocessor<dim> S_value_postprocessor_defect(boundary_values_name);
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_update, director_postprocessor_defect);
    data_out.add_data_vector(locally_relevant_update, S_value_postprocessor_defect);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(folder, filename + "-step", step,
                                        mpi_communicator, 2, 8);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::output_rhs
(const std::string folder, const std::string filename, const int step) const
{
    DirectorPostprocessor<dim>
        director_postprocessor_defect(boundary_values_name);
    SValuePostprocessor<dim> S_value_postprocessor_defect(boundary_values_name);
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(system_rhs, director_postprocessor_defect);
    data_out.add_data_vector(system_rhs, S_value_postprocessor_defect);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(folder, filename + "-rhs-", step,
                                        mpi_communicator, 2, 8);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::output_term1
(const std::string folder, const std::string filename, const int step) const
{
    DirectorPostprocessor<dim>
        director_postprocessor_defect(boundary_values_name);
    SValuePostprocessor<dim> S_value_postprocessor_defect(boundary_values_name);
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(rhs_term1, director_postprocessor_defect);
    data_out.add_data_vector(rhs_term1, S_value_postprocessor_defect);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(folder, filename + "-rhs-term1-", step,
                                        mpi_communicator, 2, 8);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::output_term2
(const std::string folder, const std::string filename, const int step) const
{
    DirectorPostprocessor<dim>
        director_postprocessor_defect(boundary_values_name);
    SValuePostprocessor<dim> S_value_postprocessor_defect(boundary_values_name);
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(rhs_term2, director_postprocessor_defect);
    data_out.add_data_vector(rhs_term2, S_value_postprocessor_defect);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(folder, filename + "-rhs-term2-", step,
                                        mpi_communicator, 2, 8);
}



template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::output_term3
(const std::string folder, const std::string filename, const int step) const
{
    DirectorPostprocessor<dim>
        director_postprocessor_defect(boundary_values_name);
    SValuePostprocessor<dim> S_value_postprocessor_defect(boundary_values_name);
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(rhs_term3, director_postprocessor_defect);
    data_out.add_data_vector(rhs_term3, S_value_postprocessor_defect);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(folder, filename + "-rhs-term3-", step,
                                        mpi_communicator, 2, 8);
}




template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::output_test
(const std::string folder, const std::string filename, const int step) const
{
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names(msc::vec_dim<dim>);
    for (int i = 0; i < msc::vec_dim<dim>; ++i)
        solution_names[i] = "component-" + std::to_string(i);
    data_out.add_data_vector(locally_relevant_solution, solution_names);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(folder, filename, step,
                                        mpi_communicator, 2, 8);
}




template <int dim, int order>
void IsoSteadyStateMPI<dim, order>::run()
{
    make_grid(num_refines,
              left_endpoint,
              right_endpoint);
    setup_system(true);

    output_results(data_folder, initial_config_filename, 0);

    unsigned int iteration = 0;
    double residual_norm{std::numeric_limits<double>::max()};

    while (residual_norm > simulation_tol && iteration < simulation_max_iters)
    {
        iteration++;
        setup_system(false);
        output_results(data_folder, final_config_filename, iteration);
        assemble_system(iteration);
        // output_rhs(data_folder, final_config_filename, iteration);
        // output_term1(data_folder, final_config_filename, iteration);
        // output_term2(data_folder, final_config_filename, iteration);
        // output_term3(data_folder, final_config_filename, iteration);
        solve();
        // output_update(data_folder, final_config_filename, iteration);
        residual_norm = system_rhs.l2_norm();
        pcout << "Residual is: " << residual_norm << "\n";

        if (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
        {
            pcout << "Outputting iteration " << iteration << " \n";
            pcout << "Residual norm is: " << residual_norm << " \n";
            dealii::TimerOutput::Scope t(computing_timer, "output");
        }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << "\n";
    }
}



#include "IsoSteadyStateMPI.inst"
