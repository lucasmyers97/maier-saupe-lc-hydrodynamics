#include "IsoTimeDependentMPI.hpp"

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
#include <deal.II/dofs/dof_renumbering.h>
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

#include "Utilities/maier_saupe_constants.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "Numerics/LagrangeMultiplier.hpp"
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
IsoTimeDependentMPI<dim, order>::IsoTimeDependentMPI(const po::variables_map &vm)
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
                      dealii::TimerOutput::cpu_and_wall_times)
    , total_iterations(0)

    , lagrange_multiplier_eff(vm["lagrange-step-size"].as<double>(),
                              vm["lagrange-tol"].as<double>(),
                              vm["lagrange-max-iters"].as<int>())
    , boundary_value_func(BoundaryValuesFactory::BoundaryValuesFactory<dim>(vm))

    , left_endpoint(vm["left-endpoint"].as<double>())
    , right_endpoint(vm["right-endpoint"].as<double>())
    , num_refines(vm["num-refines"].as<int>())
    , x_refines(vm["x-refines"].as<unsigned int>())
    , y_refines(vm["y-refines"].as<unsigned int>())
    , z_refines(vm["z-refines"].as<unsigned int>())

    , simulation_step_size(vm["simulation-step-size"].as<double>())
    , simulation_tol(vm["simulation-tol"].as<double>())
    , simulation_max_iters(vm["simulation-max-iters"].as<int>())
    , maier_saupe_alpha(vm["maier-saupe-alpha"].as<double>())
    , use_amg(vm["use-amg"].as<bool>())
    , dt(vm["dt"].as<double>())
    , n_steps(vm["n-steps"].as<int>())

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
void IsoTimeDependentMPI<dim, order>::make_grid(const unsigned int num_refines,
                                              const double left,
                                              const double right,
                                              const std::vector<unsigned int> reps)
{
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    for (int i = 0; i < dim; ++i)
    {
        p1[i] = left;
        p2[i] = right;
    }
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      reps,
                                                      p1,
                                                      p2);
    triangulation.refine_global(num_refines);

  // dealii::GridGenerator::hyper_cube(triangulation, left, right);
  // triangulation.refine_global(num_refines);

  // For 3D set top and bottom to Neumann boundaries
  // if (dim == 3)
  // {
  //   for (const auto &cell : triangulation.active_cell_iterators())
  //       if (cell->is_locally_owned())
  //           for (const auto &face : cell->face_iterators())
  //           {
  //               const auto center = face->center();
  //               if (std::fabs(center[dim - 1] - left) < 1e-12 ||
  //                   std::fabs(center[dim - 1] - right) < 1e-12)
  //                   face->set_boundary_id(1);
  //           }
  // }
}



template <int dim, int order>
void IsoTimeDependentMPI<dim, order>::setup_system(bool initial_step)
{
    dealii::TimerOutput::Scope t(computing_timer, "setup");
    if (initial_step)
    {
        dof_handler.distribute_dofs(fe);
        // dealii::DoFRenumbering::Cuthill_McKee(dof_handler);
        pcout << "Running with " << dof_handler.n_dofs() << " DoFs"
              << std::endl;

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        locally_relevant_solution.reinit(locally_owned_dofs,
                                         locally_relevant_dofs,
                                         mpi_communicator);
        locally_relevant_past_solution.reinit(locally_owned_dofs,
                                              locally_relevant_dofs,
                                              mpi_communicator);

        // Create local solution with initial condition, copy to ghosted solution vector
        LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                               mpi_communicator);
        dealii::VectorTools::interpolate(dof_handler,
                                         *boundary_value_func,
                                         locally_owned_solution);
        locally_owned_solution.compress(dealii::VectorOperation::insert);
        locally_relevant_solution = locally_owned_solution;
        locally_relevant_past_solution = locally_owned_solution;

        // Make ghosted solution continuous by using constrainsts object
        dealii::AffineConstraints<double> solution_constraints;
        solution_constraints.clear();
        solution_constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                        solution_constraints);
        dealii::VectorTools::interpolate_boundary_values(
            dof_handler,
            0,
            *boundary_value_func,
            solution_constraints);
        solution_constraints.close();
        solution_constraints.distribute(locally_relevant_solution);
        solution_constraints.distribute(locally_relevant_past_solution);
        locally_relevant_solution.compress(dealii::VectorOperation::insert);
        locally_relevant_past_solution.compress(dealii::VectorOperation::insert);

        // Set constrants on system update to 0 on the boundaries
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::interpolate_boundary_values(
            dof_handler,
            0,
            dealii::Functions::ZeroFunction<dim>(msc::vec_dim<dim>),
            constraints);
        constraints.close();
    }

    // Make the sparsity pattern
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                       dof_handler.locally_owned_dofs(),
                                                       mpi_communicator,
                                                       locally_relevant_dofs);

    // if (initial_step)
    // {
    //     dealii::SparsityPattern sparsity_pattern;
    //     sparsity_pattern.copy_from(dsp);

    //     std::ofstream out("sparsity_pattern.svg");
    //     sparsity_pattern.print_svg(out);
    // }

    // Set up system rhs and matrix
    system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs,
                      mpi_communicator);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}




template <int dim, int order>
void IsoTimeDependentMPI<dim, order>::assemble_system(int step)
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

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        old_solution_gradients
        (n_q_points,
         std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        old_solution_values(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        previous_solution_values(n_q_points, dealii::Vector<double>(fe.components));
    dealii::Vector<double> Lambda(fe.components);
    dealii::FullMatrix<double> R(fe.components, fe.components);
    std::vector<dealii::Vector<double>>
        R_inv_phi(dofs_per_cell, dealii::Vector<double>(fe.components));
    double shape_value = 0;

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);
            fe_values.get_function_gradients(locally_relevant_solution,
                                             old_solution_gradients);
            fe_values.get_function_values(locally_relevant_solution,
                                          old_solution_values);
            fe_values.get_function_values(locally_relevant_past_solution,
                                          previous_solution_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                Lambda = 0;
                R = 0;

                lagrange_multiplier_eff.invertQ(old_solution_values[q]);
                lagrange_multiplier_eff.returnLambda(Lambda);
                lagrange_multiplier_eff.returnJac(R);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const unsigned int component_j =
                        fe.system_to_component_index(j).first;
                    R_inv_phi[j].reinit(fe.components);
                    shape_value = fe_values.shape_value(j, q);
                    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
                        R_inv_phi[j][i] = R[i][component_j] * shape_value;
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
        } // if_locally_owned loop
        // cell_idx++;
    } // cell loop

    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim, int order>
void IsoTimeDependentMPI<dim, order>::solve(const bool use_amg)
{
    dealii::TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-10);

    LA::SolverGMRES solver(solver_control, mpi_communicator);

    if (use_amg)
    {
        LA::MPI::PreconditionAMG preconditioner;
        LA::MPI::PreconditionAMG::AdditionalData data;
        data.symmetric_operator = false;
        preconditioner.initialize(system_matrix, data);
        solver.solve(system_matrix,
                     completely_distributed_solution,
                     system_rhs,
                     preconditioner);
    } else
    {
      dealii::PETScWrappers::PreconditionNone preconditioner;
      preconditioner.initialize(system_matrix);
      solver.solve(system_matrix,
                   completely_distributed_solution,
                   system_rhs,
                   preconditioner);
    }
    total_iterations += solver_control.last_step();
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
double IsoTimeDependentMPI<dim, order>::determine_step_length()
{
  return simulation_step_size;
}



template <int dim, int order>
void IsoTimeDependentMPI<dim, order>::output_results
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
void IsoTimeDependentMPI<dim, order>::
iterate_timestep(const int current_timestep)
{
    unsigned int iterations = 0;
    double residual_norm{std::numeric_limits<double>::max()};

    // solves system and puts solution in `current_solution` variable
    while (residual_norm > simulation_tol && iterations < simulation_max_iters)
    {
        setup_system(false);
        assemble_system(current_timestep);
        solve(false);
        residual_norm = system_rhs.l2_norm();
        pcout << "Residual is: " << residual_norm << std::endl;
        pcout << "Norm of newton update is: " << locally_relevant_solution.l2_norm()
                  << std::endl;
        iterations++;
    }

    if (residual_norm > simulation_tol) {
        std::terminate();
    }

    LA::MPI::Vector completely_distributed_configuration(locally_owned_dofs,
                                                         mpi_communicator);
    completely_distributed_configuration = locally_relevant_solution;
    locally_relevant_past_solution = completely_distributed_configuration;
}



template <int dim, int order>
void IsoTimeDependentMPI<dim, order>::run()
{
    pcout << "Running simulation in " << dim << "D" << std::endl;

    std::vector<unsigned int> reps(dim);
    reps[0] = x_refines;
    reps[1] = y_refines;
    if (dim == 3)
        reps[2] = z_refines;
    make_grid(num_refines,
              left_endpoint,
              right_endpoint,
              reps);
    setup_system(true);

    output_results(data_folder, initial_config_filename, 0);

    for (int current_step = 1; current_step < n_steps; ++current_step)
    {
      pcout << "Running timestep" << current_step << "\n";
      iterate_timestep(current_step);

      if (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          output_results(data_folder, final_config_filename, current_step);

      pcout << "\n\n";
    }

    computing_timer.print_summary();
    computing_timer.reset();
    pcout << "\n";
}



#include "IsoTimeDependentMPI.inst"
