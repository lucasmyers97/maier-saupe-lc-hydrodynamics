#include "HydroFixedConfiguration.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/numerics/vector_tools_boundary.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

#include "Utilities/maier_saupe_constants.hpp"
#include "Utilities/SimulationOptions.hpp"
#include "LiquidCrystalSystems/IsoTimeDependent.hpp"
#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "Numerics/LagrangeMultiplier.hpp"
#include "Numerics/InverseMatrix.hpp"
#include "Numerics/SchurComplement.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
HydroFixedConfiguration<dim>::
HydroFixedConfiguration(const unsigned int degree_,
                        const dealii::Triangulation<dim> &triangulation_,
                        const double zeta_1_,
                        const double zeta_2_)
    : fe(dealii::FE_Q<dim>(degree_ + 1), dim, dealii::FE_Q<dim>(degree_), 1)
    , dof_handler(triangulation_)
    , degree(degree_)
    , zeta_1(zeta_1_)
    , zeta_2(zeta_2_)
{}



template <int dim>
void HydroFixedConfiguration<dim>::setup_dofs()
{
    A_preconditioner.reset();
    system_matrix.clear();
    preconditioner_matrix.clear();

    dof_handler.distribute_dofs(fe);
    dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);

    {
        constraints.clear();

        dealii::FEValuesExtractors::Vector velocities(0);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler,
                                                         0,
                                                         dealii::Functions::ZeroFunction<dim>(dim + 1),
                                                         constraints,
                                                         fe.component_mask(velocities));
    }

    constraints.close();

    const std::vector<dealii::types::global_dof_index> dofs_per_block =
        dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    {
        dealii::BlockDynamicSparsityPattern dsp(2, 2);

        dsp.block(0, 0).reinit(n_u, n_u);
        dsp.block(1, 0).reinit(n_p, n_u);
        dsp.block(0, 1).reinit(n_u, n_p);
        dsp.block(1, 1).reinit(n_p, n_p);

        dsp.collect_sizes();

        dealii::Table<2, dealii::DoFTools::Coupling> coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
            for (unsigned int d = 0; d < dim + 1; ++d)
                if (!((c == dim) && (d == dim)))
                    coupling[c][d] = dealii::DoFTools::always;
                else
                    coupling[c][d] = dealii::DoFTools::none;

        dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                coupling,
                                                dsp,
                                                constraints,
                                                false);

        sparsity_pattern.copy_from(dsp);
    }

    {
        dealii::BlockDynamicSparsityPattern preconditioner_dsp(2, 2);

        preconditioner_dsp.block(0, 0).reinit(n_u, n_u);
        preconditioner_dsp.block(1, 0).reinit(n_p, n_u);
        preconditioner_dsp.block(0, 1).reinit(n_u, n_p);
        preconditioner_dsp.block(1, 1).reinit(n_p, n_p);

        preconditioner_dsp.collect_sizes();

        dealii::Table<2, dealii::DoFTools::Coupling>
            preconditioner_coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
            for (unsigned int d = 0; d < dim + 1; ++d)
                if (((c == dim) && (d == dim)))
                    preconditioner_coupling[c][d] = dealii::DoFTools::always;
                else
                    preconditioner_coupling[c][d] = dealii::DoFTools::none;

        dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                preconditioner_coupling,
                                                preconditioner_dsp,
                                                constraints,
                                                false);

        preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    }

    system_matrix.reinit(sparsity_pattern);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

    solution.reinit(2);
    solution.block(0).reinit(n_u);
    solution.block(1).reinit(n_p);
    solution.collect_sizes();

    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_u);
    system_rhs.block(1).reinit(n_p);
    system_rhs.collect_sizes();
}



template <int dim>
void HydroFixedConfiguration<dim>::
assemble_system(const std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                &stress_tensor,
                const std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                &Q_tensor)
{
    system_matrix         = 0;
    system_rhs            = 0;
    preconditioner_matrix = 0;
    dealii::QGauss<dim> quadrature_formula(degree + 2);
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
                for (unsigned int j = 0; j <= i; ++j)
                {
                    local_matrix(i, j) +=
                        (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (1)
                         + zeta_1 * dealii::scalar_product
                                    (symgrad_phi_u[i],
                                     Q_tensor_vals[q] * symgrad_phi_u[j]
                                     - symgrad_phi_u[j] * Q_tensor_vals[q])
                         - div_phi_u[i] * phi_p[j]                 // (2)
                         - phi_p[i] * div_phi_u[j])                // (3)
                        * fe_values.JxW(q);                        // * dx

                    local_preconditioner_matrix(i, j) +=
                        (phi_p[i] * phi_p[j]) // (4)
                        * fe_values.JxW(q);   // * dx
                }
                const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                local_rhs(i) -= (dealii::scalar_product(
                                 fe_values[velocities].gradient(i, q),
                                 stress_tensor_vals[q])
                                 * fe_values.JxW(q));
                local_rhs(i) -= (dealii::scalar_product(
                                 grad_phi_u[i],
                                 stress_tensor_vals[q] * Q_tensor_vals[q]
                                 - Q_tensor_vals[q] * stress_tensor_vals[q])
                                 * fe_values.JxW(q)
                                 * zeta_2);
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            {
                local_matrix(i, j) = local_matrix(j, i);
                local_preconditioner_matrix(i, j) =
                    local_preconditioner_matrix(j, i);
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


template <int dim>
void HydroFixedConfiguration<dim>::solve()
{
    A_preconditioner =
        std::make_shared<typename InnerPreconditioner<dim>::type>();
    A_preconditioner->initialize(system_matrix.block(0, 0),
                                 typename InnerPreconditioner<dim>::type::AdditionalData());

    const InverseMatrix<dealii::SparseMatrix<double>,
                        typename InnerPreconditioner<dim>::type>
        A_inverse(system_matrix.block(0, 0), *A_preconditioner);
    dealii::Vector<double> tmp(solution.block(0).size());

    {
        dealii::Vector<double> schur_rhs(solution.block(1).size());
        A_inverse.vmult(tmp, system_rhs.block(0));
        system_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= system_rhs.block(1);

        SchurComplement<typename InnerPreconditioner<dim>::type>
            schur_complement(system_matrix, A_inverse);

        dealii::SolverControl solver_control(solution.block(1).size(),
                                             1e-6 * schur_rhs.l2_norm());
        dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

        dealii::SparseILU<double> preconditioner;
        preconditioner.initialize(preconditioner_matrix.block(1, 1),
                                  dealii::SparseILU<double>::AdditionalData());

        InverseMatrix<dealii::SparseMatrix<double>, dealii::SparseILU<double>>
            m_inverse(preconditioner_matrix.block(1, 1), preconditioner);

        cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);

        constraints.distribute(solution);

        std::cout << "  " << solver_control.last_step()
                  << " outer CG Schur complement iterations for pressure"
                  << std::endl;
    }

    {
        system_matrix.block(0, 1).vmult(tmp, solution.block(1));
        tmp *= -1;
        tmp += system_rhs.block(0);

        A_inverse.vmult(solution.block(0), tmp);

        constraints.distribute(solution);
    }
}



template <int dim>
void HydroFixedConfiguration<dim>::solve_entire_block()
{
    dealii::SolverControl solver_control(solution.block(0).size(), 1e-10);
    // dealii::SolverCG<dealii::BlockVector<double>> cg(solver_control);
    // cg.solve(system_matrix, solution, system_rhs, dealii::PreconditionIdentity());
    // dealii::SolverGMRES<dealii::BlockVector<double>> gmres(solver_control);
    // gmres.solve(system_matrix, solution, system_rhs, dealii::PreconditionIdentity());
    dealii::SparseDirectUMFPACK sparse_direct;
    sparse_direct.initialize(system_matrix);
    sparse_direct.solve(system_rhs);
    solution = system_rhs;
}



template <int dim>
void
HydroFixedConfiguration<dim>::output_results() const
{
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim,
                                      dealii::DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output("flow_solution.vtu");
    data_out.write_vtu(output);
}

template class HydroFixedConfiguration<2>;
template class HydroFixedConfiguration<3>;
