#include "IsoTimeDependent.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/fe_field_function.h>
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

namespace po = boost::program_options;
namespace msc = maier_saupe_constants;



template <int dim, int order>
IsoTimeDependent<dim, order>::IsoTimeDependent(const po::variables_map &vm)
    : dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(2), msc::vec_dim<dim>)
    , boundary_value_func(BoundaryValuesFactory::BoundaryValuesFactory<dim>(vm))
    , lagrange_multiplier(vm["lagrange-step-size"].as<double>(),
                          vm["lagrange-tol"].as<double>(),
                          vm["lagrange-max-iters"].as<int>())

    , left_endpoint(vm["left-endpoint"].as<double>())
    , right_endpoint(vm["right-endpoint"].as<double>())
    , num_refines(vm["num-refines"].as<int>())

    , simulation_step_size(vm["simulation-step-size"].as<double>())
    , simulation_tol(vm["simulation-tol"].as<double>())
    , simulation_max_iters(vm["simulation-max-iters"].as<int>())
    , maier_saupe_alpha(vm["maier-saupe-alpha"].as<double>())
    , dt(vm["dt"].as<double>())
    , n_steps(vm["n-steps"].as<int>())

    , boundary_values_name(vm["boundary-values-name"].as<std::string>())
    , S_value(vm["S-value"].as<double>())
    , defect_charge_name(vm["defect-charge-name"].as<std::string>())

    , data_folder(vm["data-folder"].as<std::string>())
    , initial_config_filename(vm["initial-config-filename"].as<std::string>())
    , final_config_filename(vm["final-config-filename"].as<std::string>())
    , archive_filename(vm["archive-filename"].as<std::string>())
{}



template <int dim, int order>
IsoTimeDependent<dim, order>::IsoTimeDependent()
    : dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(2), msc::vec_dim<dim>)
    , lagrange_multiplier(1.0, 1e-8, 10)
{}



template <int dim, int order>
void IsoTimeDependent<dim, order>::make_grid(const unsigned int num_refines,
                                           const double left,
                                           const double right)
{
    dealii::GridGenerator::hyper_cube(triangulation, left, right);
    triangulation.refine_global(num_refines);
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::setup_system(bool initial_step)
{
    if (initial_step)
    {
        dof_handler.distribute_dofs(fe);
        current_solution.reinit(dof_handler.n_dofs());
        past_solution.reinit(dof_handler.n_dofs());

        dealii::AffineConstraints<double> configuration_constraints;
        configuration_constraints.clear();
        dealii::DoFTools::
            make_hanging_node_constraints(dof_handler,
                                          configuration_constraints);
        dealii::VectorTools::
            interpolate_boundary_values(dof_handler,
                                        /* boundary_component = */0,
                                        *boundary_value_func,
                                        configuration_constraints);
        configuration_constraints.close();

        dealii::VectorTools::project(dof_handler,
                                     configuration_constraints,
                                     dealii::QGauss<dim>(fe.degree + 1),
                                     *boundary_value_func,
                                     current_solution);
        past_solution = current_solution;
    }

    hanging_node_constraints.clear();
    dealii::DoFTools::
        make_hanging_node_constraints(dof_handler, hanging_node_constraints);
    dealii::VectorTools::
        interpolate_boundary_values(dof_handler,
                                    /* boundary_component = */0,
                                    dealii::Functions::ZeroFunction<dim>(),
                                    hanging_node_constraints);
    hanging_node_constraints.close();

    system_update.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);

    hanging_node_constraints.condense(dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::assemble_system()
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
        hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                            cell_rhs,
                                                            local_dof_indices,
                                                            system_matrix,
                                                            system_rhs);
    }
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::solve()
{
    dealii::SolverControl solver_control(5000);
    dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control);

    solver.solve(system_matrix, system_update, system_rhs,
                 dealii::PreconditionIdentity());
    hanging_node_constraints.distribute(system_update);

    const double newton_alpha = determine_step_length();
    current_solution.add(newton_alpha, system_update);
}



template <int dim, int order>
double IsoTimeDependent<dim, order>::determine_step_length()
{
  return simulation_step_size;
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::output_results
(const std::string folder, const std::string filename, const int time_step) const
{
    NematicPostprocessor<dim> nematic_postprocessor;
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, nematic_postprocessor);
    data_out.build_patches();

    std::cout << "Outputting results" << std::endl;

    std::ofstream output(folder + "-"
                         + std::to_string(time_step) + "-"
                         + filename + ".vtu");
    data_out.write_vtu(output);
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::iterate_timestep()
{
    setup_system(/*initial_timestep = */false);

    unsigned int iterations = 0;
    double residual_norm{std::numeric_limits<double>::max()};
    while (residual_norm > simulation_tol && iterations < simulation_max_iters)
    {
        assemble_system();
        solve();
        residual_norm = system_rhs.l2_norm();
        std::cout << "Residual is: " << residual_norm << std::endl;
        std::cout << "Norm of newton update is: " << system_update.l2_norm()
                  << std::endl;
    }

    if (residual_norm > simulation_tol)
        std::terminate();

    past_solution = current_solution;
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::run()
{
    make_grid(num_refines,
              left_endpoint,
              right_endpoint);
    setup_system(true);

    auto start = std::chrono::high_resolution_clock::now();
    for (int current_step = 1; current_step < n_steps; ++current_step)
    {
        std::cout << "Running timestep " << current_step << "\n";
        iterate_timestep();
        output_results(data_folder, final_config_filename, current_step);
        std::cout << "\n\n";
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "total time for solving is: "
              << duration.count() << " seconds" << std::endl;
}



template <int dim, int order>
dealii::Functions::FEFieldFunction<dim>
    IsoTimeDependent<dim, order>::return_fe_field()
{
    return dealii::Functions::FEFieldFunction<dim>(dof_handler,
                                                   current_solution);
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::output_grid(const std::string folder,
                                             const std::string filename) const
{
    std::ofstream out(filename);
    dealii::GridOut grid_out;
    grid_out.write_svg(triangulation, out);
    std::cout << "Grid written to " << filename << std::endl;
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::output_sparsity_pattern
    (const std::string folder, const std::string filename) const
{
    std::ofstream out(folder + filename);
    sparsity_pattern.print_svg(out);
}



template <int dim, int order>
void IsoTimeDependent<dim, order>::write_to_grid
    (const std::string grid_filename,
     const std::string output_filename,
     const std::vector<std::string> meshgrid_names,
     const double dist_scale) const
{
    EvaluateFEObject<dim> e_fe_o(meshgrid_names);
    e_fe_o.read_grid(grid_filename, dist_scale);
    e_fe_o.read_fe_at_points(dof_handler, current_solution);
    e_fe_o.write_values_to_grid(output_filename);
}



#include "IsoTimeDependent.inst"
