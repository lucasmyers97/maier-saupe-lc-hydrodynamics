#ifndef HYDRO_FIXED_CONFIGURATION_HPP
#define HYDRO_FIXED_CONFIGURATION_HPP

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/base/parameter_handler.h>

#include <memory>
#include <tuple>

#include "BoundaryValues/BoundaryValues.hpp"

template <int dim>
struct InnerPreconditioner;

template <>
struct InnerPreconditioner<2>
{
    using type = dealii::SparseDirectUMFPACK;
};

template <>
struct InnerPreconditioner<3>
{
    using type = dealii::SparseILU<double>;
};



template <int dim>
class HydroFixedConfiguration
{
public:
    HydroFixedConfiguration(const dealii::Triangulation<dim> &triangulation_,
                            const unsigned int degree = 1,
                            const double zeta_1_ = 1.0,
                            const double zeta_2_ = 1.0);

    void declare_parameters(dealii::ParameterHandler &prm);
    void get_parameters(dealii::ParameterHandler &prm);

    void setup_dofs();
    void assemble_system(const std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                         &stress_tensor,
                         const std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                         &Q_tensor);
    void solve();
    void solve_entire_block();
    void output_results() const;

    std::tuple<unsigned int, double, double> return_parameters() const;
    const dealii::DoFHandler<dim>& return_dof_handler() const;
    const dealii::FESystem<dim>& return_fe() const;
    const dealii::AffineConstraints<double>& return_constraints() const;
    dealii::BlockSparseMatrix<double>& return_system_matrix();
    dealii::BlockSparseMatrix<double>& return_preconditioner_matrix();
    dealii::BlockVector<double>& return_system_rhs();

private:

    int degree;

    double zeta_1 = 1.0;
    double zeta_2 = 1.0;

    dealii::FESystem<dim>      fe;
    dealii::DoFHandler<dim>    dof_handler;

    dealii::AffineConstraints<double> constraints;
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    dealii::BlockSparsityPattern      sparsity_pattern;
    dealii::BlockSparseMatrix<double> system_matrix;

    dealii::BlockSparsityPattern      preconditioner_sparsity_pattern;
    dealii::BlockSparseMatrix<double> preconditioner_matrix;

    dealii::BlockVector<double> solution;
    dealii::BlockVector<double> system_rhs;

    std::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
};

#endif
