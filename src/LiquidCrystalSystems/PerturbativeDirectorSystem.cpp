#include "PerturbativeDirectorSystem.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/hdf5.h>
 
#include <deal.II/base/types.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <limits>
 
namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA
 
 
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
 
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
 
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/grid_out.h>
 
#include <fstream>
#include <iostream>

#include "Numerics/SetDefectBoundaryConstraints.hpp"
#include "Utilities/GridTools.hpp"

template <int dim>
PerturbativeDirectorSystem<dim>::
PerturbativeDirectorSystem(unsigned int degree,
                           double left,
                           double right,
                           unsigned int num_refines,
                           unsigned int num_further_refines,
                           const std::vector<dealii::Point<dim>> &defect_pts,
                           const std::vector<double> &defect_refine_distances,
                           double defect_radius,
                           bool fix_defects,
                           const std::string &h5_filename,
                           const std::string &dataset_name,
                           const GridTools::RadialPointSet<dim> &point_set,
                           unsigned int refinement_level,
                           bool allow_merge,
                           unsigned int max_boxes,
                           BoundaryCondition boundary_condition,
                           std::unique_ptr<PerturbativeDirectorRighthandSide<dim>> righthand_side)
    : left(left)
    , right(right)
    , num_refines(num_refines)
    , num_further_refines(num_further_refines)
    , defect_pts(defect_pts)
    , defect_refine_distances(defect_refine_distances)
    , defect_radius(defect_radius)
    , fix_defects(fix_defects)

    , h5_filename(h5_filename)
    , dataset_name(dataset_name)
    , point_set(point_set)
    , refinement_level(refinement_level)
    , allow_merge(allow_merge)
    , max_boxes(max_boxes)

    , boundary_condition(boundary_condition)
    , righthand_side(std::move(righthand_side))
    , mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening))
    , fe(degree)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
{}



template <int dim>
void PerturbativeDirectorSystem<dim>::make_grid()
{
    dealii::GridGenerator::hyper_cube(triangulation, left, right);

    coarse_tria.copy_triangulation(triangulation);
    triangulation.refine_global(num_refines);

    refine_further();
    refine_around_defects();

    std::vector<dealii::types::material_id> defect_ids(defect_pts.size());
    for (std::size_t i = 0; i < defect_pts.size(); ++i)
        defect_ids[i] = i + 1;
    SetDefectBoundaryConstraints::mark_defect_domains(triangulation,
                                                      defect_pts,
                                                      defect_ids,
                                                      defect_radius);
}



/** DIMENSIONALLY-DEPENDENT but can easily be made independent */
template <int dim>
void PerturbativeDirectorSystem<dim>::refine_further()
{
    dealii::Point<dim> grid_center;
    dealii::Point<dim> cell_center;
    dealii::Point<dim> grid_cell_difference;
    double cell_distance = 0;

    grid_center[0] = 0.5 * (left + right);
    grid_center[1] = grid_center[0];

    // each refine region is half the size of the previous
    std::vector<double> refine_distances(num_further_refines);
    for (std::size_t i = 0; i < num_further_refines; ++i)
        refine_distances[i] = std::pow(0.5, i + 1) * (right - grid_center[0]);
    
    // refine each extra refinement zone
    for (const auto &refine_distance : refine_distances)
    {
        for (auto &cell : triangulation.active_cell_iterators())
        {
            cell_center = cell->center();
            grid_cell_difference = grid_center - cell_center;
            
            cell_distance = std::max(std::abs(grid_cell_difference[0]), 
                                     std::abs(grid_cell_difference[1]));

            if (cell_distance < refine_distance)
                cell->set_refine_flag();
        }

        triangulation.execute_coarsening_and_refinement();
    }
}



/** DIMENSIONALLY-DEPENDENT dependent on defects being points, 
 * could probably be made to be dimensionally-independent but it might be
 * better to just do a gradient-based adaptive refinement */
template <int dim>
void PerturbativeDirectorSystem<dim>
::refine_around_defects()
{
    dealii::Point<dim> defect_cell_difference;
    double defect_cell_distance = 0;

    for (const auto &refine_dist : defect_refine_distances)
    {
        for (const auto &defect_pt : defect_pts)
            for (auto &cell : triangulation.active_cell_iterators())
            {
                if (!cell->is_locally_owned())
                    continue;

                defect_cell_difference = defect_pt - cell->center();
                defect_cell_distance = defect_cell_difference.norm();

                if (defect_cell_distance <= refine_dist)
                    cell->set_refine_flag();
            }

        triangulation.execute_coarsening_and_refinement();
    }
}




template <int dim>
void PerturbativeDirectorSystem<dim>::setup_system()
{
    dealii::TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    if (boundary_condition == BoundaryCondition::Dirichlet)
        dealii::VectorTools::
            interpolate_boundary_values(dof_handler,
                                        0,
                                        dealii::Functions::ZeroFunction<dim>(),
                                        constraints);
    else
    {
        std::map<dealii::types::global_dof_index, dealii::Point<dim>> support_points;
        dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<dim>(),
                                                     dof_handler,
                                                     support_points);

        dealii::Point<dim> origin;
        double min_value = std::numeric_limits<double>::max();
        dealii::types::global_dof_index min_idx = 0;
        for (const auto& point : support_points)
            if (point.second.distance(origin) < min_value)
            {
                min_idx = point.first;
                min_value = point.second.distance(origin);
            }

        double global_min = dealii::Utilities::MPI::min(min_value, mpi_communicator);
        if (min_value == global_min)
            constraints.add_line(min_idx);

        constraints.make_consistent_in_parallel(locally_owned_dofs, 
                                                locally_relevant_dofs, 
                                                mpi_communicator);
    }

    // fix defects
    if (fix_defects) {
        std::map<dealii::types::material_id, const dealii::Function<dim>*>
            function_map;

        dealii::Functions::ZeroFunction<dim> homogeneous_dirichlet_function;
        for (dealii::types::material_id i = 1; i <= defect_pts.size(); ++i)
            function_map[i] = &homogeneous_dirichlet_function;

        std::map<dealii::types::global_dof_index, double> boundary_values;

        SetDefectBoundaryConstraints::
            interpolate_boundary_values(dof_handler, 
                                        function_map, 
                                        boundary_values);

        for (const auto &boundary_value : boundary_values)
        {
            if (constraints.can_store_line(boundary_value.first) &&
                !constraints.is_constrained(boundary_value.first))
            {
              constraints.add_line(boundary_value.first);
              constraints.set_inhomogeneity(boundary_value.first,
                                            boundary_value.second);
            }
        }
        constraints.make_consistent_in_parallel(locally_owned_dofs, 
                                                locally_relevant_dofs, 
                                                mpi_communicator);
    }
    constraints.close();

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
}




template <int dim>
void PerturbativeDirectorSystem<dim>::assemble_system()
{
    dealii::TimerOutput::Scope t(computing_timer, "assembly");

    const dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values | 
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points | 
                                    dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> rhs_vals(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        cell_matrix = 0.;
        cell_rhs    = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            righthand_side->value_list(fe_values.get_quadrature_points(),
                                       rhs_vals);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                       fe_values.shape_grad(j, q_point) *
                                       fe_values.JxW(q_point);

                cell_rhs(i) += rhs_vals[q_point] *                         
                               fe_values.shape_value(i, q_point) * 
                               fe_values.JxW(q_point);
            }
        }

        cell->get_dof_indices(local_dof_indices);
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
void PerturbativeDirectorSystem<dim>::solve()
{
    dealii::TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

    LA::SolverCG solver(solver_control);

    LA::MPI::PreconditionAMG preconditioner;

    LA::MPI::PreconditionAMG::AdditionalData data;

    preconditioner.initialize(system_matrix, data);

    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
}




template <int dim>
void PerturbativeDirectorSystem<dim>::refine_grid()
{
    dealii::TimerOutput::Scope t(computing_timer, "refine");

    dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    dealii::KellyErrorEstimator<dim>::
        estimate(dof_handler,
                 dealii::QGauss<dim - 1>(fe.degree + 1),
                 std::map<dealii::types::boundary_id, const dealii::Function<dim> *>(),
                 locally_relevant_solution,
                 estimated_error_per_cell);
    dealii::parallel::distributed::GridRefinement::
        refine_and_coarsen_fixed_number(triangulation, 
                                        estimated_error_per_cell, 
                                        0.3, 
                                        0.03);
    triangulation.execute_coarsening_and_refinement();
}




template <int dim>
void PerturbativeDirectorSystem<dim>::output_results(const unsigned int cycle) const
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "theta_c");

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
}



template <int dim>
void PerturbativeDirectorSystem<dim>::output_rhs() const
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(system_rhs, "rhs");

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "system_rhs", 0, mpi_communicator, 2, 8);
}



template <int dim>
void PerturbativeDirectorSystem<dim>::output_points_to_hdf5() const
{
    std::vector<hsize_t> dataset_dims = {point_set.n_r * point_set.n_theta, 
                                         fe.n_components()};

    dealii::HDF5::File file(h5_filename,
                            dealii::HDF5::File::FileAccessMode::create,
                            mpi_communicator);
    auto dataset = file.create_dataset<double>(dataset_name, dataset_dims);

    dataset.set_attribute<unsigned int>("n_r", point_set.n_r);
    dataset.set_attribute<unsigned int>("n_theta", point_set.n_theta);
    dataset.set_attribute<double>("r_0", point_set.r_0);
    dataset.set_attribute<double>("r_f", point_set.r_f);

    auto cache = dealii::GridTools::Cache<dim>(triangulation);

    auto bounding_boxes = GridTools::get_bounding_boxes<dim>(triangulation,
                                                             refinement_level,
                                                             allow_merge,
                                                             max_boxes);
    auto global_bounding_boxes 
        = dealii::GridTools::
          exchange_local_bounding_boxes(bounding_boxes, mpi_communicator);

    std::vector<double> solution_values;
    std::vector<hsize_t> solution_indices;

    std::tie(solution_values, solution_indices)
        = GridTools::read_configuration_at_radial_points(point_set,
                                                         mpi_communicator,
                                                         dof_handler,
                                                         locally_relevant_solution,
                                                         cache,
                                                         global_bounding_boxes);
    if (solution_values.empty())
        dataset.write_none<double>();
    else
        dataset.write_selection(solution_values, solution_indices);
}



template <int dim>
void PerturbativeDirectorSystem<dim>::output_cores_to_hdf5() const
{
    std::string core_filename = "./temp-data/carter-numerical-solution/core_structure.h5";
    std::string pos_dataset_name = "pos_phi";
    std::string neg_dataset_name = "neg_phi";
    GridTools::RadialPointSet<dim> pos_point_set;
    GridTools::RadialPointSet<dim> neg_point_set;

    pos_point_set.center = defect_pts[0];
    pos_point_set.r_0 = 0.1;
    pos_point_set.r_f = 30.0;
    pos_point_set.n_r = 1000;
    pos_point_set.n_theta = 1000;
    neg_point_set.center = defect_pts[1];
    neg_point_set.r_0 = 0.1;
    neg_point_set.r_f = 30.0;
    neg_point_set.n_r = 1000;
    neg_point_set.n_theta = 1000;

    std::vector<hsize_t> pos_dataset_dims = {pos_point_set.n_r * pos_point_set.n_theta, 
                                             fe.n_components()};
    std::vector<hsize_t> neg_dataset_dims = {neg_point_set.n_r * neg_point_set.n_theta, 
                                             fe.n_components()};

    dealii::HDF5::File file(core_filename,
                            dealii::HDF5::File::FileAccessMode::create,
                            mpi_communicator);
    auto pos_dataset = file.create_dataset<double>(pos_dataset_name, pos_dataset_dims);
    auto neg_dataset = file.create_dataset<double>(neg_dataset_name, neg_dataset_dims);

    pos_dataset.set_attribute<unsigned int>("n_theta", pos_point_set.n_theta);
    pos_dataset.set_attribute<unsigned int>("n_r", pos_point_set.n_r);
    pos_dataset.set_attribute<double>("r_0", pos_point_set.r_0);
    pos_dataset.set_attribute<double>("r_f", pos_point_set.r_f);
    neg_dataset.set_attribute<unsigned int>("n_theta", neg_point_set.n_theta);
    neg_dataset.set_attribute<unsigned int>("n_r", neg_point_set.n_r);
    neg_dataset.set_attribute<double>("r_0", neg_point_set.r_0);
    neg_dataset.set_attribute<double>("r_f", neg_point_set.r_f);

    auto cache = dealii::GridTools::Cache<dim>(triangulation);

    auto bounding_boxes = GridTools::get_bounding_boxes<dim>(triangulation,
                                                             refinement_level,
                                                             allow_merge,
                                                             max_boxes);
    auto global_bounding_boxes 
        = dealii::GridTools::
          exchange_local_bounding_boxes(bounding_boxes, mpi_communicator);

    std::vector<double> pos_solution_values;
    std::vector<hsize_t> pos_solution_indices;

    std::tie(pos_solution_values, pos_solution_indices)
        = GridTools::read_configuration_at_radial_points(pos_point_set,
                                                         mpi_communicator,
                                                         dof_handler,
                                                         locally_relevant_solution,
                                                         cache,
                                                         global_bounding_boxes);
    if (pos_solution_values.empty())
        pos_dataset.write_none<double>();
    else
        pos_dataset.write_selection(pos_solution_values, pos_solution_indices);

    std::vector<double> neg_solution_values;
    std::vector<hsize_t> neg_solution_indices;

    std::tie(neg_solution_values, neg_solution_indices)
        = GridTools::read_configuration_at_radial_points(neg_point_set,
                                                         mpi_communicator,
                                                         dof_handler,
                                                         locally_relevant_solution,
                                                         cache,
                                                         global_bounding_boxes);
    if (neg_solution_values.empty())
        neg_dataset.write_none<double>();
    else
        neg_dataset.write_selection(neg_solution_values, neg_solution_indices);
}




template <int dim>
void PerturbativeDirectorSystem<dim>::run()
{
    pcout << "Running with "
          << "Trilinos"
          << " on " << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    make_grid();
    std::ofstream out("grid.svg");
    dealii::GridOut grid_out;
    grid_out.write_svg(triangulation, out);
    setup_system();

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    assemble_system();
    output_rhs();
    solve();

    {
        dealii::TimerOutput::Scope t(computing_timer, "output");
        output_results(0);
        output_points_to_hdf5();
        output_cores_to_hdf5();
    }

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
}



template <int dim>
double PerturbativeDirectorRighthandSide<dim>::
value(const dealii::Point<dim> &p, const unsigned int component) const
{
    assert(defect_points.size() == 2 && "Defect points wrong size");

    std::vector<double> theta(defect_points.size());
    std::vector<double> r(defect_points.size());

    for (std::size_t i = 0; i < defect_points.size(); ++i)
    {
        auto displacement = p - defect_points[i];
        theta[i] = atan2(displacement[1], displacement[0]);
        r[i] = displacement.norm();
    }

    double q1 = defect_charges[0];
    double q2 = defect_charges[1];
    double r1 = r[0];
    double r2 = r[1];
    double theta1 = theta[0];
    double theta2 = theta[1];

    return ( q1*(2 - q1) / (r1*r1) * std::sin(2*(1 - q1)*theta1 - 2*q2*theta2)
           + q2*(2 - q2) / (r2*r2) * std::sin(2*(1 - q2)*theta2 - 2*q1*theta1)
           - 2*q1*q2 / (r1*r2) * std::sin((1 - 2*q1)*theta1 - (1 - 2*q2)*theta2) );
}



template <int dim>
void PerturbativeDirectorRighthandSide<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double> &value_list,
           const unsigned int component) const
{
    assert(defect_points.size() == 2 && "Defect points wrong size");

    double q1 = defect_charges[0];
    double q2 = defect_charges[1];

    std::vector<double> theta(defect_points.size());
    std::vector<double> r(defect_points.size());

    for (std::size_t n = 0; n < point_list.size(); ++n)
    {
        for (std::size_t i = 0; i < defect_points.size(); ++i)
        {
            auto displacement = point_list[n] - defect_points[i];
            theta[i] = atan2(displacement[1], displacement[0]);
            r[i] = displacement.norm();
        }

        double r1 = r[0];
        double r2 = r[1];
        double theta1 = theta[0];
        double theta2 = theta[1];

        value_list[n] = ( q1*(2 - q1) / (r1*r1) * std::sin(2*(1 - q1)*theta1 - 2*q2*theta2)
                        + q2*(2 - q2) / (r2*r2) * std::sin(2*(1 - q2)*theta2 - 2*q1*theta1)
                        - 2*q1*q2 / (r1*r2) * std::sin((1 - 2*q1)*theta1 - (1 - 2*q2)*theta2) );
    }
}

template class PerturbativeDirectorRighthandSide<2>;
template class PerturbativeDirectorRighthandSide<3>;

template class PerturbativeDirectorSystem<2>;
template class PerturbativeDirectorSystem<3>;
