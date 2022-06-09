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

#include <memory>

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
    HydroFixedConfiguration(const unsigned int degree,
                            const dealii::Triangulation<dim> &triangulation_);

    void setup_dofs();
    void assemble_system();
    void solve();
    void output_results() const;

private:

    double degree;

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
