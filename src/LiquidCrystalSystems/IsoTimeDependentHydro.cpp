#include "IsoTimeDependentHydro.hpp"

#include <boost/iostreams/detail/select.hpp>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <boost/program_options.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "Utilities/maier_saupe_constants.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "Numerics/LagrangeMultiplier.hpp"
#include "Postprocessors/DirectorPostprocessor.hpp"
#include "Postprocessors/SValuePostprocessor.hpp"
#include "Postprocessors/EvaluateFEObject.hpp"

#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <utility>

namespace po = boost::program_options;
namespace msc = maier_saupe_constants;


// InverseMatrix class stuff --------------------------------------------------
template <class MatrixType, class PreconditionerType>
class InverseMatrix : public dealii::Subscriptor
{
public:
    InverseMatrix(const MatrixType &        m,
                  const PreconditionerType &preconditioner);

    void vmult(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src) const;

private:
    const dealii::SmartPointer<const MatrixType>         matrix;
    const dealii::SmartPointer<const PreconditionerType> preconditioner;
};

template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix
    (const MatrixType &        m,
     const PreconditionerType &preconditioner)
    : matrix(&m)
    , preconditioner(&preconditioner)
{}

template <class MatrixType, class PreconditionerType>
void InverseMatrix<MatrixType, PreconditionerType>::vmult
    (dealii::Vector<double> &      dst,
     const dealii::Vector<double> &src) const
{
    dealii::SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
    dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

    dst = 0;

    cg.solve(*matrix, dst, src, *preconditioner);
}


// SchurComplement class stuff ------------------------------------------------
template <class PreconditionerType>
class SchurComplement : public dealii::Subscriptor
{
public:
    SchurComplement
    (const dealii::BlockSparseMatrix<double> &system_matrix,
     const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType> &A_inverse);

    void vmult(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src) const;

private:
    const dealii::SmartPointer<const dealii::BlockSparseMatrix<double>> system_matrix;
    const dealii::SmartPointer<
        const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType>>
    A_inverse;

    mutable dealii::Vector<double> tmp1, tmp2;
};



template <class PreconditionerType>
SchurComplement<PreconditionerType>::SchurComplement
    (const dealii::BlockSparseMatrix<double> &system_matrix,
     const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType> &A_inverse)
    : system_matrix(&system_matrix)
    , A_inverse(&A_inverse)
    , tmp1(system_matrix.block(1, 1).m())
    , tmp2(system_matrix.block(1, 1).m())
{}


template <class PreconditionerType>
void SchurComplement<PreconditionerType>::vmult
    (dealii::Vector<double> &      dst,
     const dealii::Vector<double> &src) const
{
    system_matrix->block(1, 2).vmult(tmp1, src);
    A_inverse->vmult(tmp2, tmp1);
    system_matrix->block(2, 1).vmult(dst, tmp2);
}



template <int dim>
class HydroBoundaryValues : public dealii::Function<dim>
{
public:
    HydroBoundaryValues(std::unique_ptr<BoundaryValues<dim>> boundary_values_)
        : dealii::Function<dim>(msc::vec_dim<dim> + dim + 1)
        , boundary_values(std::move(boundary_values_))
    {};

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        return (*boundary_values).value(p, component);
    };

    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &value) const override
    {
        (*boundary_values).vector_value(p, value);
    };

    virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                            std::vector<double> &value_list,
                            const unsigned int component = 0) const override
    {
        (*boundary_values).value_list(point_list, value_list);
    };

    virtual void vector_value_list(
        const std::vector<dealii::Point<dim>> &point_list,
        std::vector<dealii::Vector<double>> &value_list) const override
    {
        (*boundary_values).vector_value_list(point_list, value_list);
    };

private:
    std::unique_ptr<BoundaryValues<dim>> boundary_values;
};

template <int dim, int order>
IsoTimeDependentHydro<dim, order>::IsoTimeDependentHydro(const po::variables_map &vm)
    : dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(2), msc::vec_dim<dim>,
         dealii::FE_Q<dim>(2), dim,
         dealii::FE_Q<dim>(1), 1)
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
IsoTimeDependentHydro<dim, order>::IsoTimeDependentHydro()
    : dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(1), msc::vec_dim<dim>,
         dealii::FE_Q<dim>(2), dim,
         dealii::FE_Q<dim>(1), 1)
    , lagrange_multiplier(1.0, 1e-8, 10)
{}



template <int dim, int order>
void IsoTimeDependentHydro<dim, order>::make_grid(const unsigned int num_refines,
                                           const double left,
                                           const double right)
{
    dealii::GridGenerator::hyper_cube(triangulation, left, right);
    triangulation.refine_global(num_refines);
}



template <int dim, int order>
void IsoTimeDependentHydro<dim, order>::setup_system(bool initial_step)
{
    A_preconditioner.reset();
    system_matrix.clear();
    preconditioner_matrix.clear();

    dof_handler.distribute_dofs(fe);
    std::vector<unsigned int> block_component(msc::vec_dim<dim> + dim + 1, 0);
    for (unsigned int i = msc::vec_dim<dim>; i < msc::vec_dim<dim> + dim; ++i)
        block_component[i] = 1;
    block_component[msc::vec_dim<dim> + dim] = 2;
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);

    const std::vector<dealii::types::global_dof_index> dofs_per_block =
        dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_q = dofs_per_block[0];
    const unsigned int n_u = dofs_per_block[1];
    const unsigned int n_p = dofs_per_block[2];

    if (initial_step)
    {
        current_solution.reinit(3);
        current_solution.block(0).reinit(n_q);
        current_solution.block(1).reinit(n_u);
        current_solution.block(2).reinit(n_p);
        current_solution.collect_sizes();

        // Project initial configuration onto system
        {
            hanging_node_constraints.clear();
            dealii::DoFTools::make_hanging_node_constraints(
                dof_handler,
                hanging_node_constraints);
            hanging_node_constraints.close();

            dealii::VectorTools::project(
                dof_handler,
                hanging_node_constraints,
                dealii::QGauss<dim>(fe.degree + 1),
                HydroBoundaryValues<dim>(std::move(boundary_value_func)),
                current_solution);
        }

        // deal with Dirichlet boundary conditions for Q and velocity
        {
            std::vector<bool>
                zero_boundary_components(msc::vec_dim<dim> + dim + 1);
            for (unsigned int i = 0; i < zero_boundary_components.size(); ++i)
            {
                if (i < msc::vec_dim<dim> + dim)
                    zero_boundary_components[i] = true;
                else
                    zero_boundary_components[i] = false;
            }
            dealii::ComponentMask component_mask(zero_boundary_components);

            hanging_node_constraints.clear();
            dealii::DoFTools::make_hanging_node_constraints(
                dof_handler,
                hanging_node_constraints);
            hanging_node_constraints.close();

            dealii::VectorTools::interpolate_boundary_values(
                dof_handler, 0,
                dealii::Functions::ZeroFunction<dim>(dim + 1 + msc::vec_dim<dim>),
                hanging_node_constraints,
                component_mask);
        }

        // make block dynamic sparsity pattern for main problem
        {
            dealii::BlockDynamicSparsityPattern dsp(3, 3);

            dsp.block(0, 0).reinit(n_q, n_q);
            dsp.block(1, 0).reinit(n_u, n_q);
            dsp.block(2, 0).reinit(n_p, n_q);
            dsp.block(0, 1).reinit(n_q, n_u);
            dsp.block(1, 1).reinit(n_u, n_u);
            dsp.block(2, 1).reinit(n_p, n_u);
            dsp.block(0, 2).reinit(n_q, n_p);
            dsp.block(1, 2).reinit(n_u, n_p);
            dsp.block(2, 2).reinit(n_p, n_p);

            dsp.collect_sizes();

            dealii::Table<2, dealii::DoFTools::Coupling>
                coupling(dim + 1 + msc::vec_dim<dim>,
                         dim + 1 + msc::vec_dim<dim>);

            for (unsigned int c = 0; c < msc::vec_dim<dim> + dim + 1; ++c)
                for (unsigned int d = 0; d < msc::vec_dim<dim> + dim + 1; ++d)
                    if ((c < msc::vec_dim<dim>) && (d < msc::vec_dim<dim>))
                        coupling[c][d] = dealii::DoFTools::always;
                    else if ((c < msc::vec_dim<dim> + dim)
                             && (d < msc::vec_dim<dim> + dim))
                        coupling[c][d] = dealii::DoFTools::always;
                    else if ((c == msc::vec_dim<dim> + dim)
                             && (d < msc::vec_dim<dim> + dim))
                        coupling[c][d] = dealii::DoFTools::always;
                    else if ((c < msc::vec_dim<dim> + dim)
                             && (d == msc::vec_dim<dim> + dim))
                        coupling[c][d] = dealii::DoFTools::always;
                    else
                        coupling[c][d] = dealii::DoFTools::none;


            dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                    coupling,
                                                    dsp,
                                                    hanging_node_constraints,
                                                    false);

            sparsity_pattern.copy_from(dsp);
        }

        // make preconditioner sparsity matrix
        {
            dealii::BlockDynamicSparsityPattern preconditioner_dsp(3, 3);

            preconditioner_dsp.block(0, 0).reinit(n_q, n_q);
            preconditioner_dsp.block(1, 0).reinit(n_u, n_q);
            preconditioner_dsp.block(2, 0).reinit(n_p, n_q);
            preconditioner_dsp.block(0, 1).reinit(n_q, n_u);
            preconditioner_dsp.block(1, 1).reinit(n_u, n_u);
            preconditioner_dsp.block(2, 1).reinit(n_p, n_u);
            preconditioner_dsp.block(0, 2).reinit(n_q, n_p);
            preconditioner_dsp.block(1, 2).reinit(n_u, n_p);
            preconditioner_dsp.block(2, 2).reinit(n_p, n_p);

            preconditioner_dsp.collect_sizes();

            dealii::Table<2, dealii::DoFTools::Coupling>
                preconditioner_coupling(msc::vec_dim<dim> + dim + 1,
                                        msc::vec_dim<dim> + dim + 1);

            for (unsigned int c = 0; c < msc::vec_dim<dim> + dim + 1; ++c)
                for (unsigned int d = 0; d < msc::vec_dim<dim> + dim + 1; ++d)
                    if ((c == msc::vec_dim<dim> + dim)
                        && (d == msc::vec_dim<dim> + dim))
                        preconditioner_coupling[c][d] = dealii::DoFTools::always;
                    else
                        preconditioner_coupling[c][d] = dealii::DoFTools::none;

            dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                    preconditioner_coupling,
                                                    preconditioner_dsp,
                                                    hanging_node_constraints,
                                                    false);

            preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
        }
    }

    system_matrix.reinit(sparsity_pattern);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

    system_update.reinit(3);
    system_update.block(0).reinit(n_q);
    system_update.block(1).reinit(n_u);
    system_update.block(2).reinit(n_p);
    system_update.collect_sizes();

    system_rhs.reinit(3);
    system_rhs.block(0).reinit(n_q);
    system_rhs.block(1).reinit(n_u);
    system_rhs.block(2).reinit(n_p);
    system_rhs.collect_sizes();
}



template <int dim, int order>
void IsoTimeDependentHydro<dim, order>::assemble_system(const int current_timestep)
{
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    system_matrix = 0;
    preconditioner_matrix = 0;
    system_rhs = 0;

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_hessians
                                    | dealii::update_quadrature_points
                                    | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> preconditioner_cell_matrix(dofs_per_cell,
                                                          dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);

    // data structures for Q-tensor
    std::vector<std::vector<dealii::Tensor<1, dim>>>
        old_solution_gradients
        (n_q_points,
         std::vector<dealii::Tensor<1, dim, double>>(msc::vec_dim<dim>));
    std::vector<dealii::Tensor<1, dim>> old_solution_gradients_tmp(n_q_points);

    std::vector<dealii::Vector<double>>
        old_solution_values(n_q_points, dealii::Vector<double>(msc::vec_dim<dim>));
    std::vector<double> old_solution_values_tmp(n_q_points);

    std::vector<dealii::Vector<double>>
        old_solution_laplacians(n_q_points,
                                dealii::Vector<double>(msc::vec_dim<dim>));
    std::vector<double> old_solution_laplacians_tmp(n_q_points);

    std::vector<dealii::Vector<double>>
        previous_solution_values(n_q_points, dealii::Vector<double>(msc::vec_dim<dim>));
    std::vector<double> previous_solution_values_tmp(n_q_points);

    std::vector<dealii::FEValuesExtractors::Scalar> q_idx(msc::vec_dim<dim>);
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        q_idx[i] = dealii::FEValuesExtractors::Scalar(i);

    dealii::Vector<double> Q(msc::vec_dim<dim>);
    dealii::Vector<double> Lambda(msc::vec_dim<dim>);
    dealii::LAPACKFullMatrix<double> R(msc::vec_dim<dim>, msc::vec_dim<dim>);
    std::vector<dealii::Vector<double>>
        R_inv_phi(dofs_per_cell, dealii::Vector<double>(msc::vec_dim<dim>));

    // data structures for flow
    const dealii::FEValuesExtractors::Vector velocities(msc::vec_dim<dim>);
    const dealii::FEValuesExtractors::Scalar pressure(msc::vec_dim<dim> + dim);
    std::vector<dealii::SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    // data structures for flow forcing term
    dealii::SymmetricTensor<2, dim> sigma_d;
    dealii::SymmetricTensor<2, dim> H;

    // data structures for factoring flow into Q-evolution
    std::vector<dealii::Tensor<1, dim>> u_vals(n_q_points);
    std::vector<dealii::Tensor<2, dim>> u_grads(n_q_points);
    std::vector<double> u_grad_phi(dofs_per_cell);
    dealii::Vector<double> u_grad_Q(msc::vec_dim<dim>);

    dealii::Vector<double> W(3);
    dealii::Vector<double> eta_vec(msc::vec_dim<dim>);
    dealii::FullMatrix<double> eta_Jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    dealii::Vector<double> eps_vec(msc::vec_dim<dim>);
    std::vector<dealii::Vector<double>> eta_Jac_phi(dofs_per_cell);

    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        preconditioner_cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);

        // Calculate previous q-tensor values
        for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        {
            fe_values[q_idx[i]].get_function_gradients(current_solution,
                                                       old_solution_gradients_tmp);
            fe_values[q_idx[i]].get_function_values(current_solution,
                                                    old_solution_values_tmp);
            fe_values[q_idx[i]].get_function_laplacians(current_solution,
                                                        old_solution_laplacians_tmp);
            fe_values[q_idx[i]].get_function_values(previous_solution,
                                                    previous_solution_values_tmp);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                old_solution_gradients[q][i] = old_solution_gradients_tmp[q];
                old_solution_values[q][i] = old_solution_values_tmp[q];
                old_solution_laplacians[q][i] = old_solution_laplacians_tmp[q];
                previous_solution_values[q][i] = previous_solution_values_tmp[q];
            }
        }

        // Calculate previous flow-velocity values
        fe_values[velocities].get_function_values(current_solution, u_vals);
        fe_values[velocities].get_function_gradients(current_solution, u_grads);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // Calculate Q quadrature-specific values
            Q = old_solution_values[q];
            Lambda.reinit(msc::vec_dim<dim>);
            R.reinit(msc::vec_dim<dim>);
            lagrange_multiplier.invertQ(Q);
            lagrange_multiplier.returnLambda(Lambda);
            lagrange_multiplier.returnJac(R);
            R.compute_lu_factorization();

            // Calculate u quadrature-specific values
            W[0] = u_grads[q][1][0] - u_grads[q][0][1];
            W[1] = (dim == 3) ? u_grads[q][2][0] - u_grads[q][0][2] : 0;
            W[2] = (dim == 3) ? u_grads[q][2][1] - u_grads[q][1][2] : 0;

            eta_vec[0] = -2 * (Q[1]*W[0] - Q[2]*W[1]);
            eta_vec[1] = Q[0]*W[0] - Q[2]*W[2] - Q[3]*W[0] - Q[4]*W[1];
            eta_vec[2] = 2*Q[0]*W[1] + Q[1]*(W[1] + W[2]) - Q[4]*W[0];
            eta_vec[3] = 2*Q[0]*W[2] + Q[1]*(W[1] + W[2]) + Q[2]*W[0] + Q[3]*W[2];

            eta_Jac.reinit(msc::vec_dim<dim>, msc::vec_dim<dim>);
            eta_Jac[0][1] = -2*W[0];
            eta_Jac[0][2] = -2*W[1];
            eta_Jac[1][0] = W[0];
            eta_Jac[1][2] = -W[2];
            eta_Jac[1][3] = -W[0];
            eta_Jac[1][4] = -W[1];
            eta_Jac[2][0] = 2*W[1];
            eta_Jac[2][1] = W[1] + W[2];
            eta_Jac[2][4] = -W[0];
            eta_Jac[3][1] = 2*W[0];
            eta_Jac[3][4] = -2*W[2];
            eta_Jac[4][0] = W[2];
            eta_Jac[4][1] = W[1] + W[2];
            eta_Jac[4][2] = W[0];
            eta_Jac[4][3] = W[2];

            eps_vec[0] = u_grads[q][0][0];
            eps_vec[1] = 0.5*(u_grads[q][1][0] + u_grads[q][0][1]);
            eps_vec[3] = u_grads[q][1][1];
            if (dim == 3)
            {
                eps_vec[2] = 0.5 * (u_grads[q][2][0] + u_grads[q][0][2]);
                eps_vec[4] = 0.5*(u_grads[q][2][1] + u_grads[q][1][2]);
            }

            for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                u_grad_Q[k] = u_vals[q] * old_solution_gradients[q][k];

            // calculate single-index values from weak form
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int component_k =
                    fe.system_to_component_index(k).first;
                if (component_k < msc::vec_dim<dim>)
                {
                    R_inv_phi[k].reinit(msc::vec_dim<dim>);
                    R_inv_phi[k][component_k] = fe_values.shape_value(k, q);
                    R.solve(R_inv_phi[k]);

                    eta_Jac_phi[k].reinit(msc::vec_dim<dim>);
                    eta_Jac_phi[k][component_k] = fe_values.shape_value(k, q);
                    eta_Jac.vmult(eta_Jac_phi[k], eta_Jac_phi[k]);

                    u_grad_phi[k] = u_vals[q] * fe_values.shape_grad(k, q);
                } else
                {
                    symgrad_phi_u[k] =
                        fe_values[velocities].symmetric_gradient(k, q);
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);
                    phi_p[k]     = fe_values[pressure].value(k, q);
                }
            }

            // calculate hydro source terms from Q-configuration
            sigma_d.clear();
            for (unsigned int i = 0; i < dim; ++i)
                for (unsigned int j = i; j < dim; ++j)
                {
                    for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                        sigma_d[i][j] -= 2 * old_solution_gradients[q][k][i]
                                           * old_solution_gradients[q][k][j];

                    sigma_d[i][j] -= old_solution_gradients[q][0][i]
                                     * old_solution_gradients[q][3][j]
                                     +
                                     old_solution_gradients[q][0][j]
                                     * old_solution_gradients[q][3][i];
                }

            H.clear();
            H[0][0] = maier_saupe_alpha * old_solution_values[q][0]
                      + old_solution_laplacians[q][0] - Lambda[0];
            H[0][1] = maier_saupe_alpha * old_solution_values[q][1]
                      + old_solution_laplacians[q][1] - Lambda[1];
            H[0][2] = maier_saupe_alpha * old_solution_values[q][2]
                      + old_solution_laplacians[q][2] - Lambda[2];
            H[1][1] = maier_saupe_alpha * old_solution_values[q][3]
                      + old_solution_laplacians[q][3] - Lambda[3];
            if (dim == 3)
            {
                H[1][2] = maier_saupe_alpha * old_solution_values[q][4]
                          + old_solution_laplacians[q][4] - Lambda[4];
                H[2][2] = -(H[0][0] + H[1][1]);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i =
                    fe.system_to_component_index(i).first;

                // do Q-tensor matrix + rhs construction
                if (component_i < msc::vec_dim<dim>)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int component_j =
                            fe.system_to_component_index(j).first;
                        if (!(component_j < msc::vec_dim<dim>))
                            continue;

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

                        if (coupled_hydro)
                        {
                            cell_matrix(i, j) +=
                                (((component_i == component_j) ?
                                  fe_values.shape_value(i, q)
                                  * u_grad_phi[j] :
                                  0)
                                 -
                                 (fe_values.shape_value(i, q)
                                  * eta_Jac_phi[j][component_i])
                                 ) * dt * fe_values.JxW(q);
                        }
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

                    if (coupled_hydro)
                    {
                        cell_rhs(i) -=
                            ((fe_values.shape_value(i, q)
                              * u_grad_Q[component_i]
                              )
                             -
                             (fe_values.shape_value(i, q)
                              * eta_vec[component_i]
                              )
                             -
                             (fe_values.shape_value(i, q)
                              * eps_vec[component_i]
                              * gamma_1)
                             ) * dt * fe_values.JxW(q);
                    }
                } else
                {
                    for (unsigned int j = 0; j <= i; ++j)
                    {
                        const unsigned int component_j =
                            fe.system_to_component_index(j).first;
                        if (component_j < msc::vec_dim<dim>)
                            continue;

                        cell_matrix(i, j) +=
                            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (1)
                             - div_phi_u[i] * phi_p[j]                 // (2)
                             - phi_p[i] * div_phi_u[j])                // (3)
                            * fe_values.JxW(q);                        // * dx

                        preconditioner_cell_matrix(i, j) +=
                            (phi_p[i] * phi_p[j]) // (4)
                            * fe_values.JxW(q);   // * dx
                    }
                    cell_rhs(i) -= (dealii::scalar_product(
                                    fe_values[velocities].gradient(i, q),
                                    sigma_d)
                                    * 2 * mu
                                    * fe_values.JxW(q));
                    cell_rhs(i) -= (dealii::scalar_product(
                                    fe_values[velocities].gradient(i, q),
                                    H)
                                    * 2 * gamma
                                    * fe_values.JxW(q));
                }
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const unsigned int component_i =
                fe.system_to_component_index(i).first;
            if (component_i < msc::vec_dim<dim>)
                continue;

            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            {
                const unsigned int component_j =
                    fe.system_to_component_index(j).first;
                if (component_j < msc::vec_dim<dim>)
                    continue;
                cell_matrix(i, j) = cell_matrix(j, i);
                preconditioner_cell_matrix(i, j) =
                    preconditioner_cell_matrix(j, i);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        hanging_node_constraints.distribute_local_to_global(cell_matrix,
                                                            cell_rhs,
                                                            local_dof_indices,
                                                            system_matrix,
                                                            system_rhs);
        hanging_node_constraints.distribute_local_to_global(preconditioner_cell_matrix,
                                                            local_dof_indices,
                                                            preconditioner_matrix);
    }

    A_preconditioner =
        std::make_shared<typename InnerPreconditioner<dim>::type>();
    std::cout << system_matrix.block(1, 1).m() << " by "
              << system_matrix.block(1, 1).n() << std::endl;
    A_preconditioner->initialize(system_matrix.block(1, 1),
                                 typename InnerPreconditioner<dim>::type::AdditionalData());
}



template <int dim, int order>
void IsoTimeDependentHydro<dim, order>::solve()
{
    dealii::SolverControl q_solver_control(5000);
    dealii::SolverGMRES<dealii::Vector<double>> q_solver(q_solver_control);

    q_solver.solve(system_matrix.block(0, 0),
                   system_update.block(0),
                   system_rhs.block(0),
                   dealii::PreconditionIdentity());

    const InverseMatrix<dealii::SparseMatrix<double>,
                        typename InnerPreconditioner<dim>::type>
        A_inverse(system_matrix.block(1, 1), *A_preconditioner);
    dealii::Vector<double> tmp(system_update.block(1).size());
    {
        dealii::Vector<double> schur_rhs(system_update.block(2).size());
        A_inverse.vmult(tmp, system_rhs.block(1));
        system_matrix.block(2, 1).vmult(schur_rhs, tmp);
        schur_rhs -= system_rhs.block(2);

        SchurComplement<typename InnerPreconditioner<dim>::type>
            schur_complement(system_matrix, A_inverse);

        dealii::SolverControl p_solver_control(system_update.block(2).size(),
                                               1e-6 * schur_rhs.l2_norm());
        dealii::SolverCG<dealii::Vector<double>> cg(p_solver_control);

        dealii::SparseILU<double> preconditioner;
        preconditioner.initialize(preconditioner_matrix.block(2, 2),
                                  dealii::SparseILU<double>::AdditionalData());

        InverseMatrix<dealii::SparseMatrix<double>, dealii::SparseILU<double>>
            m_inverse(preconditioner_matrix.block(2, 2), preconditioner);

        cg.solve(schur_complement, system_update.block(2), schur_rhs, m_inverse);

        hanging_node_constraints.distribute(system_update);

        std::cout << "  " << p_solver_control.last_step()
                  << " outer CG Schur complement iterations for pressure"
                  << std::endl;
    }
    {
        system_matrix.block(1, 2).vmult(tmp, system_update.block(2));
        tmp *= -1;
        tmp += system_rhs.block(1);

        A_inverse.vmult(system_update.block(1), tmp);

        hanging_node_constraints.distribute(system_update);
    }

    const double newton_alpha = determine_step_length();
    current_solution.block(0).add(newton_alpha, system_update.block(0));
    current_solution.block(1) = system_update.block(1);
    current_solution.block(2) = system_update.block(2);
}



template <int dim, int order>
double IsoTimeDependentHydro<dim, order>::determine_step_length()
{
    if (!(coupled_hydro))
        return 1.0;
    else
        return simulation_step_size;
}




// template <int dim, int order>
// dealii::Functions::FEFieldFunction<dim>
//     IsoTimeDependentHydro<dim, order>::return_fe_field()
// {
//     return dealii::Functions::FEFieldFunction<dim>(dof_handler,
//                                                    current_solution);
// }



// template <int dim, int order>
// void IsoTimeDependentHydro<dim, order>::output_grid(const std::string folder,
//                                              const std::string filename) const
// {
//     std::ofstream out(filename);
//     dealii::GridOut grid_out;
//     grid_out.write_svg(triangulation, out);
//     std::cout << "Grid written to " << filename << std::endl;
// }



// template <int dim, int order>
// void IsoTimeDependentHydro<dim, order>::output_sparsity_pattern
//     (const std::string folder, const std::string filename) const
// {
//     std::ofstream out(folder + filename);
//     sparsity_pattern.print_svg(out);
// }



template <int dim, int order>
void IsoTimeDependentHydro<dim, order>::output_results
(const std::string folder, const std::string filename, const int time_step) const
{
    std::string data_name = boundary_values_name;
    DirectorPostprocessor<dim> director_postprocessor_defect(data_name);
    SValuePostprocessor<dim> S_value_postprocessor_defect(data_name);

    std::vector<std::string> solution_names(msc::vec_dim<dim> + dim + 1, " ");
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(msc::vec_dim<dim> + dim + 1,
                                      dealii::DataComponentInterpretation::component_is_scalar);
    for (unsigned int i = msc::vec_dim<dim>; i < msc::vec_dim<dim> + dim; ++i)
    {
      solution_names[i] = "velocity";
      data_component_interpretation[i]
          = dealii::DataComponentInterpretation::component_is_part_of_vector;
    }
    solution_names[msc::vec_dim<dim> + dim] = "pressure";
    data_component_interpretation[msc::vec_dim<dim> + dim] =
        dealii::DataComponentInterpretation::component_is_scalar;

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, director_postprocessor_defect);
    data_out.add_data_vector(current_solution, S_value_postprocessor_defect);
    data_out.add_data_vector(current_solution,
                             solution_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::cout << "Outputting results" << std::endl;

    std::ofstream output(folder + filename + "_"
                         + std::to_string(time_step) + ".vtu");
    data_out.write_vtu(output);

    // std::vector<std::string> solution_names;
    // solution_names.emplace_back("Q1");
    // solution_names.emplace_back("Q2");
    // solution_names.emplace_back("Q3");
    // solution_names.emplace_back("Q4");
    // solution_names.emplace_back("Q5");
    // solution_names.emplace_back("vx");
    // solution_names.emplace_back("vy");
    // solution_names.emplace_back("p");

    // std::vector<
    //     dealii::DataComponentInterpretation::DataComponentInterpretation>
    //     data_component_interpretation
    //     (msc::vec_dim<dim> + dim + 1,
    //      dealii::DataComponentInterpretation::component_is_scalar);
    // dealii::DataOut<dim> data_out1;
    // data_out1.attach_dof_handler(dof_handler);
    // data_out1.add_data_vector(current_solution, solution_names,
    //                           dealii::DataOut<dim>::type_dof_data,
    //                           data_component_interpretation);
    // data_out1.build_patches();

    // std::ofstream output1("Q-components.vtu");
    // data_out1.write_vtu(output1);
}



template <int dim, int order>
void IsoTimeDependentHydro<dim, order>::iterate_timestep(const int current_timestep)
{
    setup_system(false);
    unsigned int iterations = 0;
    double residual_norm{std::numeric_limits<double>::max()};

    // solves system and puts solution in `current_solution` variable
    while (residual_norm > simulation_tol && iterations < simulation_max_iters)
    {
        assemble_system(current_timestep);
        solve();
        residual_norm = system_rhs.block(0).l2_norm();
        std::cout << "Residual is: " << residual_norm << std::endl;
        std::cout << "Norm of newton update is: " << system_update.l2_norm()
                  << std::endl;
    }

    if (residual_norm > simulation_tol) {
        throw std::runtime_error("Q-solution did not converge");
    }

    previous_solution = current_solution;
}



template <int dim, int order>
void IsoTimeDependentHydro<dim, order>::run()
{
    make_grid(num_refines,
              left_endpoint,
              right_endpoint);
    setup_system(true);
    output_results(data_folder, initial_config_filename, 0);
    previous_solution = current_solution;

    for (int current_step = 1; current_step < n_steps; ++current_step)
    {
        std::cout << "Running timestep " << current_step << "\n";
        // if (current_step == 10)
        // {
        //     coupled_hydro = true;
        //     std::cout << "Coupling hydro now\n";
        // }
        setup_system(false);
        iterate_timestep(current_step);
        output_results(data_folder, final_config_filename, current_step);
        std::cout << "\n\n";
    }
}



#include "IsoTimeDependentHydro.inst"
