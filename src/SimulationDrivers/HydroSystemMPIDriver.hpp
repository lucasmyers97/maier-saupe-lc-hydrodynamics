#ifndef HYDRO_SYSTEM_MPI_DRIVER_HPP
#define HYDRO_SYSTEM_MPI_DRIVER_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/lac/generic_linear_algebra.h>

// namespace LA = dealii::LinearAlgebraPETSc;
namespace LA = dealii::LinearAlgebraTrilinos;

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <memory>
#include <tuple>

#include "LiquidCrystalSystems/HydroSystemMPI.hpp"
#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Utilities/Serialization.hpp"

template <int dim>
class HydroSystemMPIDriver
{
public:
    HydroSystemMPIDriver(std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                         stress_tensor_,
                         std::unique_ptr<dealii::TensorFunction<2, dim, double>>
                         Q_tensor_,
                         unsigned int num_refines_,
                         double left_,
                         double right_);

    void run();
    void run_coupled();

private:
    void make_grid();
    void assemble_coupled_system(HydroSystemMPI<dim>& hydro_system,
                                 const NematicSystemMPI<dim>& nematic_system);

    MPI_Comm mpi_communicator;

    dealii::ConditionalOStream pcout;
    dealii::TimerOutput computing_timer;

    dealii::parallel::distributed::Triangulation<dim> tria;
    std::unique_ptr<dealii::TensorFunction<2, dim, double>> stress_tensor;
    std::unique_ptr<dealii::TensorFunction<2, dim, double>> Q_tensor;
    unsigned int num_refines;
    double left;
    double right;

};

template <int dim>
HydroSystemMPIDriver<dim>::
HydroSystemMPIDriver(std::unique_ptr<dealii::TensorFunction<2, dim, double>> stress_tensor_,
                     std::unique_ptr<dealii::TensorFunction<2, dim, double>> Q_tensor_,
                     unsigned int num_refines_,
                     double left_,
                     double right_)
    : mpi_communicator(MPI_COMM_WORLD)
    , tria(mpi_communicator,
           typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      dealii::TimerOutput::summary,
                      dealii::TimerOutput::cpu_and_wall_times)
    , stress_tensor(std::move(stress_tensor_))
    , Q_tensor(std::move(Q_tensor_))
    , num_refines(num_refines_)
    , left(left_)
    , right(right_)
{}



template <int dim>
void HydroSystemMPIDriver<dim>::make_grid()
{
    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(num_refines);
}



template <int dim>
void HydroSystemMPIDriver<dim>::
assemble_coupled_system(HydroSystemMPI<dim> &hydro_system,
                        const NematicSystemMPI<dim> &nematic_system)
{
    const int order = 974;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const int max_iter = 20;
    LagrangeMultiplierAnalytic<dim> lagrange_multiplier(order,
                                                        alpha,
                                                        tol,
                                                        max_iter);

    const double maier_saupe_alpha = nematic_system.return_parameters();

    int degree;
    double zeta_1;
    double zeta_2;

    std::tie(zeta_1, zeta_2) = hydro_system.return_parameters();

    const dealii::DoFHandler<dim>& dof_handler
        = hydro_system.return_dof_handler();
    const dealii::FESystem<dim>& fe
        = hydro_system.return_fe();
    const dealii::AffineConstraints<double>& constraints
        = hydro_system.return_constraints();
    LA::MPI::BlockSparseMatrix& system_matrix
        = hydro_system.return_system_matrix();
    LA::MPI::BlockVector& system_rhs
        = hydro_system.return_system_rhs();
    LA::MPI::BlockSparseMatrix& preconditioner_matrix
        = hydro_system.return_preconditioner_matrix();

    const dealii::DoFHandler<dim>& nematic_dof_handler
        = nematic_system.return_dof_handler();
    const LA::MPI::Vector& nematic_solution
        = nematic_system.return_current_solution();

    system_matrix         = 0;
    system_rhs            = 0;
    preconditioner_matrix = 0;
    dealii::QGauss<dim> quadrature_formula(degree + 2);
    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_JxW_values |
                                    dealii::update_gradients);
    dealii::FEValues<dim> nematic_fe_values(nematic_dof_handler.get_fe(),
                                       quadrature_formula,
                                       dealii::update_values |
                                       dealii::update_hessians);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_Q_components
        = nematic_dof_handler.get_fe().n_components();


    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                           dofs_per_cell);
    dealii::Vector<double>     local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index>
        local_dof_indices(dofs_per_cell);

    const dealii::FEValuesExtractors::Vector velocities(0);
    const dealii::FEValuesExtractors::Scalar pressure(dim);
    std::vector<dealii::SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<dealii::Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    std::vector<dealii::Vector<double>>
        Q_vector_vals(n_q_points, dealii::Vector<double>(n_Q_components));
    std::vector<std::vector<dealii::Tensor<1, dim, double>>>
        Q_grad_vals(n_q_points,
                    std::vector<dealii::Tensor<1, dim, double>>
                    (n_Q_components));
    std::vector<dealii::Vector<double>>
        Q_laplace_vals(n_q_points, dealii::Vector<double>(n_Q_components));

    dealii::Tensor<2, dim, double> Q_mat;
    dealii::Vector<double> Lambda;
    dealii::SymmetricTensor<2, dim, double> H;
    dealii::SymmetricTensor<2, dim, double> sigma_d;

    auto cell = dof_handler.begin_active();
    const auto endc = dof_handler.end();
    auto lc_cell = nematic_dof_handler.begin_active();

    for (; cell != endc; ++cell, ++lc_cell)
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            nematic_fe_values.reinit(lc_cell);
            local_matrix                = 0;
            local_preconditioner_matrix = 0;
            local_rhs                   = 0;

            nematic_fe_values.get_function_values(nematic_solution,
                                                  Q_vector_vals);
            nematic_fe_values.get_function_gradients(nematic_solution,
                                                     Q_grad_vals);
            nematic_fe_values.get_function_laplacians(nematic_solution,
                                                      Q_laplace_vals);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                Lambda = 0;
                lagrange_multiplier.invertQ(Q_vector_vals[q]);
                lagrange_multiplier.returnLambda(Lambda);

                Q_mat[0][0] = Q_vector_vals[q][0];
                Q_mat[0][1] = Q_vector_vals[q][1];
                Q_mat[1][1] = Q_vector_vals[q][3];
                Q_mat[1][0] = Q_mat[0][1];
                if (dim == 3)
                {
                    Q_mat[0][2] = Q_vector_vals[q][2];
                    Q_mat[1][2] = Q_vector_vals[q][4];
                    Q_mat[2][0] = Q_mat[0][2];
                    Q_mat[2][1] = Q_mat[1][2];
                }

                sigma_d.clear();
                for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = i; j < dim; ++j)
                     {
                         for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                             sigma_d[i][j] -= 2 * Q_grad_vals[q][k][i]
                                 * Q_grad_vals[q][k][j];

                         sigma_d[i][j] -= Q_grad_vals[q][0][i]
                             * Q_grad_vals[q][3][j]
                             +
                             Q_grad_vals[q][0][j]
                             * Q_grad_vals[q][3][i];
                     }

                H.clear();
                H[0][0] = (maier_saupe_alpha * Q_vector_vals[q][0]
                           + Q_laplace_vals[q][0] - Lambda[0]);
                H[0][1] = (maier_saupe_alpha * Q_vector_vals[q][1]
                           + Q_laplace_vals[q][1] - Lambda[1]);
                H[1][1] = (maier_saupe_alpha * Q_vector_vals[q][3]
                           + Q_laplace_vals[q][3] - Lambda[3]);
                if (dim == 3)
                {
                    H[0][2] = (maier_saupe_alpha * Q_vector_vals[q][2]
                               + Q_laplace_vals[q][2] - Lambda[2]);
                    H[1][2] = (maier_saupe_alpha * Q_vector_vals[q][4]
                               + Q_laplace_vals[q][4] - Lambda[4]);
                    H[2][2] = -(H[0][0] + H[1][1]);
                }


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
                             + zeta_1 * dealii::scalar_product
                                        (symgrad_phi_u[i],
                                         Q_mat * symgrad_phi_u[j]
                                         - symgrad_phi_u[j] * Q_mat)
                             - div_phi_u[i] * phi_p[j]                 // (2)
                             - phi_p[i] * div_phi_u[j])                // (3)
                            * fe_values.JxW(q);                        // * dx

                        local_preconditioner_matrix(i, j) +=
                            (dealii::scalar_product(grad_phi_u[i],
                                                    grad_phi_u[j])
                            + phi_p[i] * phi_p[j]) // (4)
                            * fe_values.JxW(q);   // * dx
                    }

                    local_rhs(i) -= (dealii::
                                     scalar_product(grad_phi_u[i],
                                                    sigma_d)
                                     * zeta_1
                                     * fe_values.JxW(q));
                    local_rhs(i) -= (dealii::scalar_product(
                                     grad_phi_u[i],
                                     H)
                                     * zeta_2
                                     * fe_values.JxW(q));
                    local_rhs(i) -= (dealii::scalar_product(
                                     grad_phi_u[i],
                                     H * Q_mat - Q_mat * H)
                                     * zeta_2
                                     * fe_values.JxW(q));
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
void HydroSystemMPIDriver<dim>::run()
{
    unsigned int degree = 1;
    double zeta_1 = 1.0;
    double zeta_2 = 1.0;

    make_grid();
    HydroSystemMPI<dim> hydro_system(tria, degree, zeta_1, zeta_2);

    {
        dealii::TimerOutput::Scope t(computing_timer, "setup dofs");
        pcout << "setting up dofs\n";
        hydro_system.setup_dofs(mpi_communicator);
    }

    {
        dealii::TimerOutput::Scope t(computing_timer, "assemble system");
        pcout << "assembling system\n";
        hydro_system.assemble_system(stress_tensor, Q_tensor);
        hydro_system.output_rhs(mpi_communicator,
                                std::string("./"),
                                std::string("hydro_rhs"),
                                0);
    }

    {
        dealii::TimerOutput::Scope t(computing_timer, "solve");
        pcout << "solving system\n";
        hydro_system.build_block_schur_preconditioner();
        unsigned int n_iters = hydro_system.solve_block_schur(mpi_communicator);
        // int n_iters = hydro_system.solve(mpi_communicator);
        pcout << "Solved in " << n_iters << " iterations\n\n";
    }
    hydro_system.output_results(mpi_communicator,
                                tria,
                                std::string("./"),
                                std::string("hydro_test"),
                                0);
}



template <int dim>
void HydroSystemMPIDriver<dim>::run_coupled()
{
    unsigned int degree = 1;
    double zeta_1 = 1.0;
    double zeta_2 = 1.0;

    dealii::Triangulation<dim> coarse_tria;
    std::string filename("nematic_simulation_trilinos");
    std::unique_ptr<NematicSystemMPI<dim>> nematic_system
        = Serialization::deserialize_nematic_system(mpi_communicator,
                                                    filename,
                                                    degree,
                                                    coarse_tria,
                                                    tria);
    --degree;
    std::cout << "Read in serialization\n";
    HydroSystemMPI<dim> hydro_system(tria, degree, zeta_1, zeta_2);
    hydro_system.setup_dofs(mpi_communicator);
    std::cout << "Dofs set up\n";
    assemble_coupled_system(hydro_system, *nematic_system);
    std::cout << "system assembled\n";

    // int n_iterations = hydro_system.solve(mpi_communicator);
    hydro_system.build_block_schur_preconditioner();
    unsigned int n_iterations = hydro_system.solve_block_schur(mpi_communicator);
    std::cout << "system solved in : " << n_iterations << " iterations\n";
    hydro_system.output_results(mpi_communicator,
                                tria,
                                std::string("./"),
                                std::string("coupled_hydro_test"),
                                0);
}


#endif
