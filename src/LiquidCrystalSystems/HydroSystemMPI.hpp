#ifndef HYDRO_SYSTEM_MPI_HPP
#define HYDRO_SYSTEM_MPI_HPP

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA = dealii::LinearAlgebraTrilinos;

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/index_set.h>

#include <deal.II/base/parameter_handler.h>

#include <memory>
#include <tuple>
#include <vector>

#include "BoundaryValues/BoundaryValues.hpp"



template <int dim>
class HydroSystemMPI
{
public:
    HydroSystemMPI(const dealii::parallel::distributed::Triangulation<dim>
                   &triangulation_,
                   const unsigned int degree = 1,
                   const double zeta_1_ = 1.0,
                   const double zeta_2_ = 1.0);

    void declare_parameters(dealii::ParameterHandler &prm);
    void get_parameters(dealii::ParameterHandler &prm);

    void setup_dofs(const MPI_Comm &mpi_communicator);
    void assemble_system(const std::unique_ptr<
                         dealii::TensorFunction<2, dim, double>>
                         &stress_tensor,
                         const std::unique_ptr<
                         dealii::TensorFunction<2, dim, double>>
                         &Q_tensor);
    unsigned int solve_block_diagonal(MPI_Comm &mpi_communicator);
    void build_block_schur_preconditioner();
    unsigned int solve_block_schur(MPI_Comm &mpi_communicator);

    void output_results(const MPI_Comm &mpi_communicator,
                        const dealii::parallel::distributed::Triangulation<dim>
                        &triangulation,
                        const std::string folder,
                        const std::string filename,
                        const int time_step) const;

    std::tuple<double, double> return_parameters() const;
    const dealii::DoFHandler<dim>& return_dof_handler() const;
    const dealii::FESystem<dim>& return_fe() const;
    const dealii::AffineConstraints<double>& return_constraints() const;
    LA::MPI::BlockSparseMatrix& return_system_matrix();
    LA::MPI::BlockSparseMatrix& return_preconditioner_matrix();
    LA::MPI::BlockVector& return_system_rhs();

private:

    double zeta_1 = 1.0;
    double zeta_2 = 1.0;

    dealii::FESystem<dim>      fe;
    dealii::DoFHandler<dim>    dof_handler;

    std::vector<dealii::IndexSet> owned_partitioning;
    std::vector<dealii::IndexSet> relevant_partitioning;

    dealii::AffineConstraints<double> constraints;
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    LA::MPI::BlockSparseMatrix system_matrix;
    LA::MPI::BlockSparseMatrix preconditioner_matrix;

    LA::MPI::BlockVector solution;
    LA::MPI::BlockVector system_rhs;

    std::shared_ptr<LA::MPI::PreconditionJacobi> Mp_preconditioner;
    std::shared_ptr<LA::MPI::PreconditionAMG> AMG_preconditioner;
};

#endif
