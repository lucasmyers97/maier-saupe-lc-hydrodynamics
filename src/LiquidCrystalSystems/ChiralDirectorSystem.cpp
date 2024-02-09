#include "ChiralDirectorSystem.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/hdf5.h>
 
#include <deal.II/base/types.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <limits>
#include <stdexcept>
 
namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA
 
 
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
 
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
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
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/grid_out.h>
 
#include <fstream>
#include <iostream>

#include "Numerics/SetDefectBoundaryConstraints.hpp"
#include "Utilities/GridTools.hpp"
#include "Utilities/DefectGridGenerator.hpp"

template <int dim>
ChiralDirectorSystem<dim>::
ChiralDirectorSystem(unsigned int degree,

                     double alpha,
                     double zeta,

                     std::string grid_name,
                     dealii::Point<dim> grid_center,
                     double grid_radius,
                     unsigned int num_refines,
                     unsigned int num_further_refines,
                     const std::vector<dealii::Point<dim>> &defect_pts,
                     const std::vector<double> &defect_refine_distances,
                     double defect_radius,
                     std::string grid_filename,

                     ChiralDirectorSystem<dim>::SolverType solver_type,

                     const std::string data_folder,
                     const std::string solution_vtu_filename,
                     const std::string rhs_vtu_filename,
                     const std::string outer_structure_filename,
                     const std::string dataset_name,
                     const std::string core_structure_filename,
                     const std::string pos_dataset_name,
                     const std::string neg_dataset_name,

                     const GridTools::RadialPointSet<dim> &point_set,
                     unsigned int refinement_level,
                     bool allow_merge,
                     unsigned int max_boxes,
                     std::unique_ptr<dealii::Function<dim>> righthand_side,
                     std::unique_ptr<dealii::Function<dim>> boundary_function)
    : alpha(alpha)
    , zeta(zeta)
    , grid_name(grid_name)
    , grid_center(grid_center)
    , grid_radius(grid_radius)
    , num_refines(num_refines)
    , num_further_refines(num_further_refines)
    , defect_pts(defect_pts)
    , defect_refine_distances(defect_refine_distances)
    , defect_radius(defect_radius)
    , grid_filename(grid_filename)

    , solver_type(solver_type)

    , data_folder(data_folder)
    , solution_vtu_filename(solution_vtu_filename)
    , rhs_vtu_filename(rhs_vtu_filename)
    , outer_structure_filename(outer_structure_filename)
    , dataset_name(dataset_name)
    , core_structure_filename(core_structure_filename)
    , pos_dataset_name(pos_dataset_name)
    , neg_dataset_name(neg_dataset_name)

    , point_set(point_set)
    , refinement_level(refinement_level)
    , allow_merge(allow_merge)
    , max_boxes(max_boxes)

    , righthand_side(std::move(righthand_side))
    , boundary_function(std::move(boundary_function))
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
void ChiralDirectorSystem<dim>::make_grid()
{
    if (grid_name == "hyper_ball")
        dealii::GridGenerator::hyper_ball(triangulation, 
                                          grid_center, 
                                          grid_radius);
    else if (grid_name == "hyper_ball_balanced")
        dealii::GridGenerator::hyper_ball_balanced(triangulation, 
                                                   grid_center, 
                                                   grid_radius);
    else
        throw std::invalid_argument("grid_name must be either 'hyper_ball'"
                                    " or 'hyper_ball_balanced'");

    coarse_tria.copy_triangulation(triangulation);
    triangulation.refine_global(num_refines);

    refine_further();
    refine_around_defects();
}



template <int dim>
void ChiralDirectorSystem<dim>::read_grid()
{
    dealii::GridIn<dim> grid_in(triangulation);
    std::fstream ifs(grid_filename);

    grid_in.read(ifs, dealii::GridIn<dim>::Format::msh);

    double defect_distance_threshold 
        = (defect_radius + defect_pts[0].distance(defect_pts[1])/2);
    for (auto &cell : triangulation.active_cell_iterators())
        for (auto &face : cell->face_iterators())
        {
            if (!face->at_boundary())
                continue;

            if (face->center().distance(defect_pts[0]) < defect_distance_threshold)
                face->set_manifold_id(2);
            if (face->center().distance(defect_pts[1]) < defect_distance_threshold)
                face->set_manifold_id(3);
        }

    triangulation.set_manifold(2, dealii::PolarManifold<dim>(defect_pts[0]));
    triangulation.set_manifold(3, dealii::PolarManifold<dim>(defect_pts[1]));

    triangulation.refine_global(num_refines);
    refine_further();
    refine_around_defects();
}



/** DIMENSIONALLY-DEPENDENT but can easily be made independent */
template <int dim>
void ChiralDirectorSystem<dim>::refine_further()
{
    dealii::Point<dim> cell_center;
    dealii::Point<dim> grid_cell_difference;
    double cell_distance = 0;

    // each refine region is half the size of the previous
    std::vector<double> refine_distances(num_further_refines);
    for (std::size_t i = 0; i < num_further_refines; ++i)
        refine_distances[i] = std::pow(0.5, i + 1) * grid_radius;
    
    // refine each extra refinement zone
    for (const auto &refine_distance : refine_distances)
    {
        for (auto &cell : triangulation.active_cell_iterators())
        {
            cell_center = cell->center();
            grid_cell_difference = grid_center - cell_center;
            
            cell_distance = grid_cell_difference.norm();

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
void ChiralDirectorSystem<dim>
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
void ChiralDirectorSystem<dim>::setup_system()
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
    system_rhs_solution.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::VectorTools::
        interpolate_boundary_values(dof_handler,
                                    0,
                                    dealii::Functions::ZeroFunction<dim>(),
                                    constraints);

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
    mass_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
}



template <int dim>
void ChiralDirectorSystem<dim>::setup_system_direct()
{
    dealii::TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    locally_relevant_solution_direct.reinit(dof_handler.n_dofs());
    system_rhs_direct.reinit(dof_handler.n_dofs());
    system_rhs_solution_direct.reinit(dof_handler.n_dofs());

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::VectorTools::
        interpolate_boundary_values(dof_handler,
                                    0,
                                    dealii::Functions::ZeroFunction<dim>(),
                                    constraints);

    constraints.close();

    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);

    system_matrix_direct.reinit(sparsity_pattern);
    mass_matrix_direct.reinit(sparsity_pattern);
}




template <int dim>
void ChiralDirectorSystem<dim>::assemble_system()
{
    dealii::TimerOutput::Scope t(computing_timer, "assembly");

    const dealii::QGauss<dim> quadrature_formula(fe.degree + 1);
    const dealii::QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values | 
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points | 
                                    dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> rhs_vals(n_q_points);
    dealii::Vector<double> x_curl_grad_eta(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta(dofs_per_cell);
    dealii::Point<dim> x;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        cell_matrix = 0.;
        cell_mass_matrix = 0;
        cell_rhs    = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            righthand_side->value_list(fe_values.get_quadrature_points(),
                                       rhs_vals);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                grad_eta[k] = fe_values.shape_grad(k, q_point);
                x = fe_values.quadrature_point(q_point);

                x_curl_grad_eta[k] = x[0] * grad_eta[k][1] - x[1] * grad_eta[k][0];
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += (
                      (1 + zeta) * grad_eta[i] * grad_eta[j]
                      + alpha*alpha * (1 - zeta) * x_curl_grad_eta[i] * x_curl_grad_eta[j]
                      ) 
                      * 
                      fe_values.JxW(q_point);

                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                            fe_values.shape_value(j, q_point) *
                                            fe_values.JxW(q_point);
                }

                cell_rhs(i) -= alpha * alpha * (1 - zeta) * 
                               rhs_vals[q_point] *                         
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
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
    }

    system_matrix.compress(dealii::VectorOperation::add);
    mass_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void ChiralDirectorSystem<dim>::assemble_system_direct()
{
    dealii::TimerOutput::Scope t(computing_timer, "assembly");

    const dealii::QGauss<dim> quadrature_formula(fe.degree + 1);
    const dealii::QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values | 
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points | 
                                    dealii::update_JxW_values);
    dealii::FEFaceValues<dim> fe_face_values(fe,
                                             face_quadrature_formula,
                                             dealii::update_values | 
                                             dealii::update_quadrature_points |
                                             dealii::update_normal_vectors |
                                             dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> rhs_vals(n_q_points);
    dealii::Vector<double> x_curl_grad_eta(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta(dofs_per_cell);
    dealii::Point<dim> x;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        cell_matrix = 0.;
        cell_mass_matrix = 0;
        cell_rhs    = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
            righthand_side->value_list(fe_values.get_quadrature_points(),
                                       rhs_vals);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                grad_eta[k] = fe_values.shape_grad(k, q_point);
                x = fe_values.quadrature_point(q_point);

                x_curl_grad_eta[k] = x[0] * grad_eta[k][1] - x[1] * grad_eta[k][0];
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += (
                      (1 + zeta) * grad_eta[i] * grad_eta[j]
                      + alpha*alpha * (1 - zeta) * x_curl_grad_eta[i] * x_curl_grad_eta[j]
                      ) 
                      * 
                      fe_values.JxW(q_point);

                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q_point) *
                                            fe_values.shape_value(j, q_point) *
                                            fe_values.JxW(q_point);
                }

                cell_rhs(i) -= alpha * alpha * (1 - zeta) * 
                               rhs_vals[q_point] *                         
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
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
    }

    // system_matrix.compress(dealii::VectorOperation::add);
    // mass_matrix.compress(dealii::VectorOperation::add);
    // system_rhs.compress(dealii::VectorOperation::add);
}




template <int dim>
void ChiralDirectorSystem<dim>::solve()
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

    LA::MPI::Vector residual_vec(locally_owned_dofs, mpi_communicator);

    system_matrix.vmult(residual_vec, completely_distributed_solution);
    residual_vec -= system_rhs;

    pcout << "(Ax - b) residual is: " << residual_vec.l2_norm() << "\n";
}



template <int dim>
void ChiralDirectorSystem<dim>::solve_direct()
{
    dealii::TimerOutput::Scope t(computing_timer, "solve");

    dealii::SparseDirectUMFPACK solver;
    solver.factorize(system_matrix_direct);
    solver.vmult(locally_relevant_solution_direct, system_rhs_direct);

    pcout << "   Solved in " << "1" << " iterations."
          << std::endl;

    constraints.distribute(locally_relevant_solution_direct);

    dealii::Vector<double> residual_vec(dof_handler.n_dofs());

    system_matrix_direct.vmult(residual_vec, locally_relevant_solution_direct);
    residual_vec -= system_rhs_direct;

    pcout << "(Ax - b) residual is: " << residual_vec.l2_norm() << "\n";
}




template <int dim>
void ChiralDirectorSystem<dim>::solve_mass_matrix()
{
    dealii::TimerOutput::Scope t(computing_timer, "solve mass matrix");

    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

    LA::SolverCG solver(solver_control);

    LA::MPI::PreconditionAMG preconditioner;

    LA::MPI::PreconditionAMG::AdditionalData data;

    preconditioner.initialize(mass_matrix, data);

    solver.solve(mass_matrix,
                 system_rhs_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(system_rhs_solution);
}




template <int dim>
void ChiralDirectorSystem<dim>::solve_mass_matrix_direct()
{
    dealii::TimerOutput::Scope t(computing_timer, "solve mass matrix");

    dealii::SparseDirectUMFPACK solver;
    solver.factorize(mass_matrix_direct);
    solver.vmult(system_rhs_solution_direct, system_rhs_direct);

    pcout << "   Solved in " << "1" << " iterations."
          << std::endl;

    constraints.distribute(system_rhs_solution_direct);
}




template <int dim>
void ChiralDirectorSystem<dim>::output_results(const unsigned int cycle) const
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "theta_c");

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(data_folder, 
                                        solution_vtu_filename, 
                                        cycle, 
                                        mpi_communicator, 
                                        2, 
                                        8);
}



template <int dim>
void ChiralDirectorSystem<dim>::output_rhs() const
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(system_rhs_solution, "rhs");

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(data_folder, 
                                        rhs_vtu_filename, 
                                        0, 
                                        mpi_communicator, 
                                        2, 
                                        8);
}



template <int dim>
void ChiralDirectorSystem<dim>::output_points_to_hdf5() const
{
    std::vector<hsize_t> dataset_dims = {point_set.n_r * point_set.n_theta, 
                                         fe.n_components()};

    std::string h5_filename = data_folder + outer_structure_filename;
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
void ChiralDirectorSystem<dim>::output_cores_to_hdf5() const
{
    std::string core_filename = data_folder + core_structure_filename;
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
void ChiralDirectorSystem<dim>::output_archive() const
{
    std::string archive_filename = data_folder + solution_vtu_filename;

    dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
        sol_trans(dof_handler);

    sol_trans.prepare_for_serialization(locally_relevant_solution);
    triangulation.save(archive_filename + std::string(".mesh.ar"));
}




template <int dim>
void ChiralDirectorSystem<dim>::run()
{
    pcout << "Running with Trilinos on " 
          << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." 
          << std::endl;

    if (!grid_filename.empty())
        read_grid();
    else
        make_grid();

    if (solver_type == SolverType::CG)
    {
        setup_system();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        assemble_system();
        solve();
        solve_mass_matrix();
    }
    else if (solver_type == SolverType::Direct)
    {
        setup_system_direct();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        assemble_system_direct();
        solve_direct();
        solve_mass_matrix_direct();
    }
    output_rhs();

    {
        dealii::TimerOutput::Scope t(computing_timer, "output");
        output_results(0);
        output_points_to_hdf5();
        output_cores_to_hdf5();
        output_archive();
    }

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
}



template <int dim>
double ChiralDirectorRighthandSide<dim>::
value(const dealii::Point<dim> &p, const unsigned int component) const
{
    return 0.5 * ( (p[0]*p[0] + p[1]*p[1] + 0.5 * d * p[0]) /
                   ((p[0] + 0.5 * d) * (p[0] + 0.5 * d) + p[1]*p[1])
                   +
                   (p[0]*p[0] + p[1]*p[1] - 0.5 * d * p[0]) /
                   ((p[0] - 0.5 * d) * (p[0] - 0.5 * d) + p[1]*p[1])
                   );
}



template <int dim>
void ChiralDirectorRighthandSide<dim>::
value_list(const std::vector<dealii::Point<dim>> &p,
           std::vector<double> &value_list,
           const unsigned int component) const
{
    for (std::size_t n = 0; n < p.size(); ++n)
        value_list[n] =  0.5 * ( (p[n][0]*p[n][0] + p[n][1]*p[n][1] + 0.5 * d * p[n][0]) /
                                 ((p[n][0] + 0.5 * d) * (p[n][0] + 0.5 * d) + p[n][1]*p[n][1])
                                 +
                                 (p[n][0]*p[n][0] + p[n][1]*p[n][1] - 0.5 * d * p[n][0]) /
                                 ((p[n][0] - 0.5 * d) * (p[n][0] - 0.5 * d) + p[n][1]*p[n][1])
                                 );
}




template <int dim>
double ChiralDirectorBoundaryCondition<dim>::
value(const dealii::Point<dim> &p, const unsigned int component) const
{
    return std::atan2(p[1], p[0]) 
           - 0.5 * (std::atan2(p[1], p[0] + 0.5 * d) 
                    + std::atan2(p[1], p[0] - 0.5 * d));
}



template <int dim>
void ChiralDirectorBoundaryCondition<dim>::
value_list(const std::vector<dealii::Point<dim>> &p,
           std::vector<double> &value_list,
           const unsigned int component) const
{
    for (std::size_t i = 0; i < p.size(); ++i)
    value_list[i] = std::atan2(p[i][1], p[i][0]) 
                     - 0.5 * (std::atan2(p[i][1], p[i][0] + 0.5 * d) 
                              + std::atan2(p[i][1], p[i][0] - 0.5 * d));
}

template class ChiralDirectorRighthandSide<2>;
template class ChiralDirectorRighthandSide<3>;

template class ChiralDirectorSystem<2>;
template class ChiralDirectorSystem<3>;
