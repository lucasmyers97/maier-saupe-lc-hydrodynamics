#include "HydroSystemMPI.hpp"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/lac/vector_memory.h>

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/numerics/vector_tools_boundary.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <tuple>
#include <utility>

#include "Utilities/SimulationOptions.hpp"
#include "LiquidCrystalSystems/IsoTimeDependent.hpp"
#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "Numerics/LagrangeMultiplier.hpp"
#include "Numerics/InverseMatrix.hpp"
#include "Numerics/BlockDiagonalPreconditioner.hpp"
#include "Numerics/BlockSchurPreconditioner.hpp"
#include "ExampleFunctions/Step55ExactSolution.hpp"
#include "ExampleFunctions/Step55RightHandSide.hpp"



template <int dim>
HydroSystemMPI<dim>::
HydroSystemMPI(const dealii::parallel::distributed::Triangulation<dim>
               &triangulation_,
               const unsigned int degree_,
               const double zeta_1_,
               const double zeta_2_)
    : dof_handler(triangulation_)
    , fe(dealii::FE_Q<dim>(degree_ + 1), dim, dealii::FE_Q<dim>(degree_), 1)
    , zeta_1(zeta_1_)
    , zeta_2(zeta_2_)
{}



template <int dim>
void HydroSystemMPI<dim>::
declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Hydro system MPI");
    prm.declare_entry("zeta_1",
                      "1.0",
                      dealii::Patterns::Double(),
                      "zeta_1 dimensionless parameter in hydro weak form");
    prm.declare_entry("zeta_2",
                      "1.0",
                      dealii::Patterns::Double(),
                      "zeta_2 dimensionless parameter in hydro weak form");
    prm.leave_subsection();
}



template <int dim>
void HydroSystemMPI<dim>::
get_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Hydro system MPI");
    zeta_1 = prm.get_double("zeta_1");
    zeta_2 = prm.get_double("zeta_2");
    prm.leave_subsection();
}



template <int dim>
void HydroSystemMPI<dim>::setup_dofs(const MPI_Comm &mpi_communicator)
{
    dof_handler.distribute_dofs(fe);

    // renumber dofs so we have block structure
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);

    // partition locally-owned and locally-relevant dofs by blocks
    const std::vector<dealii::types::global_dof_index> dofs_per_block =
        dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    owned_partitioning.resize(2);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
    owned_partitioning[1] =
        dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_p);

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::
        extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    // set constraints
    {
        constraints.reinit(locally_relevant_dofs);

        dealii::FEValuesExtractors::Vector velocities(0);
        dealii::DoFTools::
            make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::
            interpolate_boundary_values(dof_handler,
                                        0,
                                        dealii::Functions::
                                        ZeroFunction<dim>(dim + 1),
                                        constraints,
                                        fe.component_mask(velocities));
        constraints.close();
    }

    // set system matrix coupling + sparsity pattern
    {
        system_matrix.clear();

        dealii::Table<2, dealii::DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
            for (unsigned int d = 0; d < dim + 1; ++d)
                if (!((c == dim) && (d == dim)))
                    coupling[c][d] = dealii::DoFTools::always;
                else
                    coupling[c][d] = dealii::DoFTools::none;

        dealii::BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

        dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                coupling,
                                                dsp,
                                                constraints,
                                                /*keep_constrained_dofs*/false);
        dealii::SparsityTools::
            distribute_sparsity_pattern(dsp,
                                        dof_handler.locally_owned_dofs(),
                                        mpi_communicator,
                                        locally_relevant_dofs);

        system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }

    // set preconditioner matrix coupling + sparsity pattern
    {
        preconditioner_matrix.clear();

        dealii::Table<2, dealii::DoFTools::Coupling> coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
            for (unsigned int d = 0; d < dim + 1; ++d)
                if (c == d)
                    coupling[c][d] = dealii::DoFTools::always;
                else
                    coupling[c][d] = dealii::DoFTools::none;

        dealii::BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

        dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                coupling,
                                                dsp,
                                                constraints,
                                                /*keep_constrained_dofs*/false);

        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            dealii::Utilities::MPI::
            all_gather(mpi_communicator,
                       dof_handler.locally_owned_dofs()),
            mpi_communicator, locally_relevant_dofs);

        preconditioner_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }

    solution.reinit(owned_partitioning,
                    relevant_partitioning,
                    mpi_communicator);
    system_rhs.reinit(owned_partitioning, mpi_communicator);
}



template <int dim>
void HydroSystemMPI<dim>::
assemble_system(const std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                &stress_tensor,
                const std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                &Q_tensor)
{
    system_matrix         = 0;
    system_rhs            = 0;
    preconditioner_matrix = 0;
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values |
                                    dealii::update_gradients);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();


    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                           dofs_per_cell);
    dealii::Vector<double>     local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const dealii::FEValuesExtractors::Vector velocities(0);
    const dealii::FEValuesExtractors::Scalar pressure(dim);
    std::vector<dealii::SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<dealii::Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    std::vector<dealii::Tensor<2, dim, double>> stress_tensor_vals(n_q_points);
    std::vector<dealii::Tensor<2, dim, double>> Q_tensor_vals(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            local_matrix                = 0;
            local_preconditioner_matrix = 0;
            local_rhs                   = 0;

            stress_tensor->value_list(fe_values.get_quadrature_points(),
                                      stress_tensor_vals);
            Q_tensor->value_list(fe_values.get_quadrature_points(),
                                 Q_tensor_vals);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] =
                        fe_values[velocities].symmetric_gradient(k, q);
                    grad_phi_u[k] =
                        fe_values[velocities].gradient(k, q);
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    phi_p[k]     = fe_values[pressure].value(k, q);
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        local_matrix(i, j) +=
                            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (1)
                             +
                             (zeta_1
                              * dealii::
                              scalar_product(symgrad_phi_u[i],
                                             Q_tensor_vals[q] * symgrad_phi_u[j]
                                             - symgrad_phi_u[j] * Q_tensor_vals[q]
                                             ))
                             -
                             div_phi_u[i] * phi_p[j]                 // (2)
                             -
                             phi_p[i] * div_phi_u[j])                // (3)
                            * fe_values.JxW(q);                        // * dx

                        local_preconditioner_matrix(i, j) +=
                            (dealii::scalar_product(grad_phi_u[i],
                                                    grad_phi_u[j])
                             + phi_p[i] * phi_p[j])// (4)
                            * fe_values.JxW(q);   // * dx
                    }

                    local_rhs(i) -= (dealii::
                                     scalar_product(grad_phi_u[i],
                                                    stress_tensor_vals[q]
                                                    )
                                     * fe_values.JxW(q));
                    local_rhs(i) -= (dealii::
                                     scalar_product(grad_phi_u[i],
                                                    (stress_tensor_vals[q]
                                                     * Q_tensor_vals[q])
                                                    -
                                                    (Q_tensor_vals[q]
                                                     * stress_tensor_vals[q]))
                                     * fe_values.JxW(q)
                                     * zeta_2);
                }
            }

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
            constraints.distribute_local_to_global(local_preconditioner_matrix,
                                                   local_dof_indices,
                                                   preconditioner_matrix);
        }
    }
    system_matrix.compress(dealii::VectorOperation::add);
    preconditioner_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
unsigned int HydroSystemMPI<dim>::
solve_block_diagonal(MPI_Comm &mpi_communicator)
{
    // set up preconditioners for blocks
    LA::MPI::PreconditionAMG prec_A;
    {
        LA::MPI::PreconditionAMG::AdditionalData data;
        prec_A.initialize(system_matrix.block(0, 0), data);
    }
    LA::MPI::PreconditionAMG prec_S;
    {
        LA::MPI::PreconditionAMG::AdditionalData data;
        prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }

    // set up block diagonal preconditioner
    using MpInverseT = InverseMatrix<LA::MPI::SparseMatrix,
                                     LA::MPI::PreconditionAMG>;
    const MpInverseT mp_inverse(preconditioner_matrix.block(1, 1), prec_S);
    const BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
                                      MpInverseT,
                                      LA::MPI::BlockVector>
        preconditioner(prec_A, mp_inverse);

    // solve
    dealii::SolverControl solver_control(system_matrix.m(),
                                         1e-10 * system_rhs.l2_norm());
    dealii::SolverFGMRES<LA::MPI::BlockVector> solver(solver_control);

    LA::MPI::BlockVector distributed_solution(owned_partitioning,
                                              mpi_communicator);
    constraints.set_zero(distributed_solution);
    solver.solve(system_matrix,
                 distributed_solution,
                 system_rhs,
                 preconditioner);
    constraints.distribute(distributed_solution);

    solution = distributed_solution;

    return solver_control.last_step();
}



template <int dim>
void HydroSystemMPI<dim>::build_block_schur_preconditioner()
{
    // constant modes are needed data for amg preconditioner
    std::vector<std::vector<bool>> constant_modes;
    const dealii::FEValuesExtractors::Vector velocity_components(0);
    dealii::DoFTools::
        extract_constant_modes(dof_handler,
                               fe.component_mask(velocity_components),
                               constant_modes);

    Mp_preconditioner = std::make_shared<LA::MPI::PreconditionJacobi>();
    AMG_preconditioner = std::make_shared<LA::MPI::PreconditionAMG>();

    LA::MPI::PreconditionAMG::AdditionalData AMG_data;
    AMG_data.constant_modes        = constant_modes;
    AMG_data.elliptic              = true;
    AMG_data.higher_order_elements = true;
    AMG_data.smoother_sweeps       = 2;
    AMG_data.aggregation_threshold = 0.02;

    Mp_preconditioner->initialize(preconditioner_matrix.block(1, 1));
    AMG_preconditioner->initialize(preconditioner_matrix.block(0, 0), AMG_data);
}



template <int dim>
unsigned int HydroSystemMPI<dim>::solve_block_schur(MPI_Comm &mpi_communicator)
{
    unsigned int n_iterations = 0;
    const double  solver_tolerance = 1e-8 * system_rhs.l2_norm();
    dealii::SolverControl solver_control(/*n_iters = */10, solver_tolerance);
    dealii::PrimitiveVectorMemory<LA::MPI::BlockVector> mem;

    LA::MPI::BlockVector distributed_solution(owned_partitioning,
                                              mpi_communicator);

    // try to solve with single AMG V-cycle for velocity matrix
    try
    {
        const BlockSchurPreconditioner<LA::MPI::PreconditionAMG,
                                       LA::MPI::PreconditionJacobi,
                                       LA::MPI::BlockSparseMatrix,
                                       LA::MPI::BlockVector,
                                       LA::MPI::Vector>
            preconditioner(system_matrix,
                           preconditioner_matrix,
                           *Mp_preconditioner,
                           *AMG_preconditioner,
                           false);

        dealii::SolverFGMRES<LA::MPI::BlockVector>::AdditionalData
            additional_data(/*max_basis_size= */50);
        dealii::SolverFGMRES<LA::MPI::BlockVector> solver(solver_control,
                                                          mem,
                                                          additional_data);
        solver.solve(system_matrix,
                     distributed_solution,
                     system_rhs,
                     preconditioner);

        n_iterations = solver_control.last_step();
    }
    // otherwise preconditioner with complete inversoin of velocity matrix
    catch (dealii::SolverControl::NoConvergence &)
    {
        const BlockSchurPreconditioner<LA::MPI::PreconditionAMG,
                                       LA::MPI::PreconditionJacobi,
                                       LA::MPI::BlockSparseMatrix,
                                       LA::MPI::BlockVector,
                                       LA::MPI::Vector>
            preconditioner(system_matrix,
                           preconditioner_matrix,
                           *Mp_preconditioner,
                           *AMG_preconditioner,
                           true);

        dealii::SolverControl solver_control_refined(system_matrix.m(),
                                                     solver_tolerance);
        dealii::SolverFGMRES<LA::MPI::BlockVector>::AdditionalData
            additional_data(/*restart_parameter= */50);
        dealii::SolverFGMRES<LA::MPI::BlockVector>
            solver(solver_control_refined, mem, additional_data);
        solver.solve(system_matrix,
                     distributed_solution,
                     system_rhs,
                     preconditioner);

        n_iterations =
          (solver_control.last_step() + solver_control_refined.last_step());
    }

    solution = distributed_solution;
    return n_iterations;
}



template <int dim>
void HydroSystemMPI<dim>::
output_results(const MPI_Comm &mpi_communicator,
               const dealii::parallel::distributed::Triangulation<dim>
               &triangulation,
               const std::string folder,
               const std::string filename,
               const int time_step) const
{
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<dealii::DataComponentInterpretation::
                DataComponentInterpretation>
        data_component_interp(dim,
                              dealii::DataComponentInterpretation::
                              component_is_part_of_vector);
    data_component_interp.push_back(dealii::DataComponentInterpretation::
                                    component_is_scalar);

    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interp);
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
std::tuple<double, double> HydroSystemMPI<dim>::
return_parameters() const
{
    return std::make_tuple(zeta_1, zeta_2);
}



template <int dim>
const dealii::DoFHandler<dim>& HydroSystemMPI<dim>::
    return_dof_handler() const
{
    return dof_handler;
}



template <int dim>
const dealii::FESystem<dim>& HydroSystemMPI<dim>::
    return_fe() const
{
    return fe;
}



template <int dim>
const dealii::AffineConstraints<double> &HydroSystemMPI<dim>::
    return_constraints() const
{
    return constraints;
}



template <int dim>
LA::MPI::BlockSparseMatrix &HydroSystemMPI<dim>::
    return_system_matrix()
{
    return system_matrix;
}



template <int dim>
LA::MPI::BlockVector &HydroSystemMPI<dim>::
    return_system_rhs()
{
    return system_rhs;
}



template <int dim>
LA::MPI::BlockSparseMatrix &HydroSystemMPI<dim>::
    return_preconditioner_matrix()
{
    return preconditioner_matrix;
}



template class HydroSystemMPI<2>;
template class HydroSystemMPI<3>;
