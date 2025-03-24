#include "nematic_energy.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <stdexcept>

#include "Numerics/LagrangeMultiplierAnalytic.hpp"

namespace nematic_energy
{
namespace LA = dealii::LinearAlgebraTrilinos;

template <>
void singular_potential_energy<2>(const MPI_Comm &mpi_communicator, 
                                  double current_time,
                                  double alpha, double B, double L2, double L3,
                                  const dealii::DoFHandler<2> &dof_handler,
                                  const LA::MPI::Vector &current_solution,
                                  LagrangeMultiplierAnalytic<2> &singular_potential,
                                  std::vector<std::vector<double>> &energy_vals)
{
    constexpr int dim = 2;

    const dealii::FESystem<dim> fe = dof_handler.get_fe();
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_hessians
                                    | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    double Z = 0;

    double mean_field_term = 0;
    double entropy_term = 0;
    double cubic_term = 0;
    double L1_elastic_term = 0;
    double L2_elastic_term = 0;
    double L3_elastic_term = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if ( !(cell->is_locally_owned()) )
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_gradients(current_solution, dQ);
        fe_values.get_function_values(current_solution, Q_vec);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            Lambda_vec = 0;

            singular_potential.invertQ(Q_vec[q]);
            singular_potential.returnLambda(Lambda_vec);
            Z = singular_potential.returnZ();

            mean_field_term += 
                (alpha*(-(Q_vec[q][0]) * (Q_vec[q][0]) 
                        - Q_vec[q][0]*Q_vec[q][3] 
                        - (Q_vec[q][1]) * (Q_vec[q][1]) 
                        - (Q_vec[q][2]) * (Q_vec[q][2]) 
                        - (Q_vec[q][3]) * (Q_vec[q][3]) 
                        - (Q_vec[q][4]) * (Q_vec[q][4])))
                * fe_values.JxW(q);

            cubic_term +=
                (3*B*(-(Q_vec[q][0]) * (Q_vec[q][0])*Q_vec[q][3] 
                      + Q_vec[q][0]*(Q_vec[q][1]) * (Q_vec[q][1]) 
                      - Q_vec[q][0]*(Q_vec[q][3]) * (Q_vec[q][3]) 
                      - Q_vec[q][0]*(Q_vec[q][4]) * (Q_vec[q][4]) 
                      + (Q_vec[q][1]) * (Q_vec[q][1])*Q_vec[q][3] 
                      + 2*Q_vec[q][1]*Q_vec[q][2]*Q_vec[q][4] 
                      - (Q_vec[q][2]) * (Q_vec[q][2])*Q_vec[q][3])
                ) * fe_values.JxW(q);
            
            entropy_term +=
                (2*Q_vec[q][0]*Lambda_vec[0] + Q_vec[q][0]*Lambda_vec[3] 
                 + 2*Q_vec[q][1]*Lambda_vec[1] + 2*Q_vec[q][2]*Lambda_vec[2] 
                 + Q_vec[q][3]*Lambda_vec[0] + 2*Q_vec[q][3]*Lambda_vec[3] 
                 + 2*Q_vec[q][4]*Lambda_vec[4] - std::log(Z) + std::log(4*M_PI))
                * fe_values.JxW(q);
            
            L1_elastic_term +=
                ((1.0/2.0)*(-dQ[q][0][0] - dQ[q][3][0]) 
                 * (-dQ[q][0][0] - dQ[q][3][0]) 
                 + (1.0/2.0)*(-dQ[q][0][1] - dQ[q][3][1]) 
                 * (-dQ[q][0][1] - dQ[q][3][1]) 
                 + (1.0/2.0)*(dQ[q][0][0]) 
                 * (dQ[q][0][0]) 
                 + (1.0/2.0)*(dQ[q][0][1]) * (dQ[q][0][1]) 
                 + (dQ[q][1][0]) * (dQ[q][1][0]) 
                 + (dQ[q][1][1]) * (dQ[q][1][1]) 
                 + (dQ[q][2][0]) * (dQ[q][2][0]) 
                 + (dQ[q][2][1]) * (dQ[q][2][1]) 
                 + (1.0/2.0)*(dQ[q][3][0]) * (dQ[q][3][0]) 
                 + (1.0/2.0)*(dQ[q][3][1]) * (dQ[q][3][1]) 
                 + (dQ[q][4][0]) * (dQ[q][4][0]) 
                 + (dQ[q][4][1]) * (dQ[q][4][1]))
                * fe_values.JxW(q);
            
            L2_elastic_term +=
                ((1.0/2.0)*L2
                 * ((dQ[q][0][0] + dQ[q][1][1]) * (dQ[q][0][0] + dQ[q][1][1]) 
                 + (dQ[q][1][0] + dQ[q][3][1]) * (dQ[q][1][0] + dQ[q][3][1]) 
                 + (dQ[q][2][0] + dQ[q][4][1]) * (dQ[q][2][0] + dQ[q][4][1])))
                * fe_values.JxW(q);
            
            L3_elastic_term +=
                ((1.0/2.0)*L3
                 *(2*((-dQ[q][0][0] - dQ[q][3][0])*(-dQ[q][0][1] - dQ[q][3][1]) 
                         + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] 
                         + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] 
                         + 2*dQ[q][4][0]*dQ[q][4][1])*Q_vec[q][1] 
                     + ((-dQ[q][0][0] - dQ[q][3][0]) 
                         * (-dQ[q][0][0] - dQ[q][3][0]) 
                         + (dQ[q][0][0]) * (dQ[q][0][0]) 
                         + 2*(dQ[q][1][0]) * (dQ[q][1][0]) 
                         + 2*(dQ[q][2][0]) * (dQ[q][2][0]) 
                         + (dQ[q][3][0]) * (dQ[q][3][0]) 
                         + 2*(dQ[q][4][0]) * (dQ[q][4][0]))*Q_vec[q][0] 
                     + ((-dQ[q][0][1] - dQ[q][3][1]) 
                         * (-dQ[q][0][1] - dQ[q][3][1]) 
                         + (dQ[q][0][1]) * (dQ[q][0][1]) 
                         + 2*(dQ[q][1][1]) * (dQ[q][1][1]) 
                         + 2*(dQ[q][2][1]) * (dQ[q][2][1]) 
                         + (dQ[q][3][1]) * (dQ[q][3][1]) 
                         + 2*(dQ[q][4][1]) * (dQ[q][4][1]))*Q_vec[q][3]))
                  * fe_values.JxW(q);
        }
    }

    double total_mean_field_term
        = dealii::Utilities::MPI::sum(mean_field_term, mpi_communicator);
    double total_cubic_term
        = dealii::Utilities::MPI::sum(cubic_term, mpi_communicator);
    double total_entropy_term
        = dealii::Utilities::MPI::sum(entropy_term, mpi_communicator);
    double total_L1_elastic_term
        = dealii::Utilities::MPI::sum(L1_elastic_term, mpi_communicator);
    double total_L2_elastic_term
        = dealii::Utilities::MPI::sum(L2_elastic_term, mpi_communicator);
    double total_L3_elastic_term
        = dealii::Utilities::MPI::sum(L3_elastic_term, mpi_communicator);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        energy_vals[0].push_back(current_time);
        energy_vals[1].push_back(total_mean_field_term);
        energy_vals[2].push_back(total_cubic_term);
        energy_vals[3].push_back(total_entropy_term);
        energy_vals[4].push_back(total_L1_elastic_term);
        energy_vals[5].push_back(total_L2_elastic_term);
        energy_vals[6].push_back(total_L3_elastic_term);
    }
}



template <>
void singular_potential_energy<3>(const MPI_Comm &mpi_communicator, 
                                  double current_time,
                                  double alpha, double B, double L2, double L3,
                                  const dealii::DoFHandler<3> &dof_handler,
                                  const LA::MPI::Vector &current_solution,
                                  LagrangeMultiplierAnalytic<3> &singular_potential,
                                  std::vector<std::vector<double>> &energy_vals)
{
    constexpr int dim = 3;

    const dealii::FESystem<dim> fe = dof_handler.get_fe();
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_hessians
                                    | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points,
           std::vector<dealii::Tensor<1, dim, double>>(fe.components));
    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));

    dealii::Vector<double> Lambda_vec(fe.components);
    double Z = 0;

    double mean_field_term = 0;
    double entropy_term = 0;
    double cubic_term = 0;
    double L1_elastic_term = 0;
    double L2_elastic_term = 0;
    double L3_elastic_term = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if ( !(cell->is_locally_owned()) )
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_gradients(current_solution, dQ);
        fe_values.get_function_values(current_solution, Q_vec);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            Lambda_vec = 0;

            singular_potential.invertQ(Q_vec[q]);
            singular_potential.returnLambda(Lambda_vec);
            Z = singular_potential.returnZ();

            mean_field_term += 
                (alpha*(-(Q_vec[q][0]) * (Q_vec[q][0]) 
                        - Q_vec[q][0]*Q_vec[q][3] 
                        - (Q_vec[q][1]) * (Q_vec[q][1]) 
                        - (Q_vec[q][2]) * (Q_vec[q][2]) 
                        - (Q_vec[q][3]) * (Q_vec[q][3]) 
                        - (Q_vec[q][4]) * (Q_vec[q][4])))                
                * fe_values.JxW(q);
            

            cubic_term +=
                (3*B*(-(Q_vec[q][0]) * (Q_vec[q][0])*Q_vec[q][3] 
                      + Q_vec[q][0]*(Q_vec[q][1]) * (Q_vec[q][1]) 
                      - Q_vec[q][0]*(Q_vec[q][3]) * (Q_vec[q][3]) 
                      - Q_vec[q][0]*(Q_vec[q][4]) * (Q_vec[q][4]) 
                      + (Q_vec[q][1]) * (Q_vec[q][1])*Q_vec[q][3] 
                      + 2*Q_vec[q][1]*Q_vec[q][2]*Q_vec[q][4] 
                      - (Q_vec[q][2]) * (Q_vec[q][2])*Q_vec[q][3])
                ) * fe_values.JxW(q);
            
            entropy_term +=
                (2*Q_vec[q][0]*Lambda_vec[0] + Q_vec[q][0]*Lambda_vec[3] 
                 + 2*Q_vec[q][1]*Lambda_vec[1] + 2*Q_vec[q][2]*Lambda_vec[2] 
                 + Q_vec[q][3]*Lambda_vec[0] + 2*Q_vec[q][3]*Lambda_vec[3] 
                 + 2*Q_vec[q][4]*Lambda_vec[4] - std::log(Z) + std::log(4*M_PI))
                * fe_values.JxW(q);
            
            L1_elastic_term +=
                (0.5*(-dQ[q][0][0] - dQ[q][3][0]) * (-dQ[q][0][0] - dQ[q][3][0]) 
                 + 0.5*(-dQ[q][0][1] - dQ[q][3][1]) * (-dQ[q][0][1] - dQ[q][3][1]) 
                 + 0.5*(-dQ[q][0][2] - dQ[q][3][2]) * (-dQ[q][0][2] - dQ[q][3][2]) 
                 + 0.5*(dQ[q][0][0]) * (dQ[q][0][0]) + 0.5*(dQ[q][0][1]) * (dQ[q][0][1]) 
                 + 0.5*(dQ[q][0][2]) * (dQ[q][0][2]) + (dQ[q][1][0]) * (dQ[q][1][0]) 
                 + (dQ[q][1][1]) * (dQ[q][1][1]) + (dQ[q][1][2]) * (dQ[q][1][2]) 
                 + (dQ[q][2][0]) * (dQ[q][2][0]) + (dQ[q][2][1]) * (dQ[q][2][1]) 
                 + (dQ[q][2][2]) * (dQ[q][2][2]) + (1.0/2.0)*(dQ[q][3][0]) * (dQ[q][3][0]) 
                 + 0.5*(dQ[q][3][1]) * (dQ[q][3][1]) + 0.5*(dQ[q][3][2]) * (dQ[q][3][2]) 
                 + (dQ[q][4][0]) * (dQ[q][4][0]) + (dQ[q][4][1]) * (dQ[q][4][1]) 
                 + (dQ[q][4][2]) * (dQ[q][4][2])) 
                * fe_values.JxW(q);
            
            L2_elastic_term +=
                (0.5*L2
                 *((dQ[q][0][0] + dQ[q][1][1] + dQ[q][2][2]) 
                     * (dQ[q][0][0] + dQ[q][1][1] + dQ[q][2][2]) 
                     + (dQ[q][1][0] + dQ[q][3][1] + dQ[q][4][2]) 
                     * (dQ[q][1][0] + dQ[q][3][1] + dQ[q][4][2]) 
                     + (-dQ[q][0][2] + dQ[q][2][0] - dQ[q][3][2] + dQ[q][4][1]) 
                     * (-dQ[q][0][2] + dQ[q][2][0] - dQ[q][3][2] + dQ[q][4][1])))  
                * fe_values.JxW(q);
            
            L3_elastic_term +=
                (0.5*L3
                 *((-Q_vec[q][0] - Q_vec[q][3])*((-dQ[q][0][2] - dQ[q][3][2]) * (-dQ[q][0][2] - dQ[q][3][2]) 
                         + (dQ[q][0][2]) * (dQ[q][0][2]) 
                         + 2*(dQ[q][1][2]) * (dQ[q][1][2]) 
                         + 2*(dQ[q][2][2]) * (dQ[q][2][2]) 
                         + (dQ[q][3][2]) * (dQ[q][3][2]) 
                         + 2*(dQ[q][4][2]) * (dQ[q][4][2])) 
                     + 2*((-dQ[q][0][0] - dQ[q][3][0])*(-dQ[q][0][1] - dQ[q][3][1]) 
                         + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] 
                         + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] 
                         + 2*dQ[q][4][0]*dQ[q][4][1])*Q_vec[q][1] 
                     + 2*((-dQ[q][0][0] - dQ[q][3][0])*(-dQ[q][0][2] - dQ[q][3][2]) 
                         + dQ[q][0][0]*dQ[q][0][2] + 2*dQ[q][1][0]*dQ[q][1][2] 
                         + 2*dQ[q][2][0]*dQ[q][2][2] + dQ[q][3][0]*dQ[q][3][2] 
                         + 2*dQ[q][4][0]*dQ[q][4][2])*Q_vec[q][2] 
                     + 2*((-dQ[q][0][1] - dQ[q][3][1])*(-dQ[q][0][2] - dQ[q][3][2]) 
                         + dQ[q][0][1]*dQ[q][0][2] + 2*dQ[q][1][1]*dQ[q][1][2] 
                         + 2*dQ[q][2][1]*dQ[q][2][2] + dQ[q][3][1]*dQ[q][3][2] 
                         + 2*dQ[q][4][1]*dQ[q][4][2])*Q_vec[q][4] 
                     + ((-dQ[q][0][0] - dQ[q][3][0]) * (-dQ[q][0][0] - dQ[q][3][0]) 
                         + (dQ[q][0][0]) * (dQ[q][0][0]) + 2*(dQ[q][1][0]) * (dQ[q][1][0]) 
                         + 2*(dQ[q][2][0]) * (dQ[q][2][0]) + (dQ[q][3][0]) * (dQ[q][3][0]) 
                         + 2*(dQ[q][4][0]) * (dQ[q][4][0]))*Q_vec[q][0] 
                     + ((-dQ[q][0][1] - dQ[q][3][1]) * (-dQ[q][0][1] - dQ[q][3][1]) 
                             + (dQ[q][0][1]) * (dQ[q][0][1]) + 2*(dQ[q][1][1]) * (dQ[q][1][1]) 
                             + 2*(dQ[q][2][1]) * (dQ[q][2][1]) + (dQ[q][3][1]) * (dQ[q][3][1]) 
                             + 2*(dQ[q][4][1]) * (dQ[q][4][1]))*Q_vec[q][3]))   
                * fe_values.JxW(q);
        }
    }

    double total_mean_field_term
        = dealii::Utilities::MPI::sum(mean_field_term, mpi_communicator);
    double total_cubic_term
        = dealii::Utilities::MPI::sum(cubic_term, mpi_communicator);
    double total_entropy_term
        = dealii::Utilities::MPI::sum(entropy_term, mpi_communicator);
    double total_L1_elastic_term
        = dealii::Utilities::MPI::sum(L1_elastic_term, mpi_communicator);
    double total_L2_elastic_term
        = dealii::Utilities::MPI::sum(L2_elastic_term, mpi_communicator);
    double total_L3_elastic_term
        = dealii::Utilities::MPI::sum(L3_elastic_term, mpi_communicator);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        energy_vals[0].push_back(current_time);
        energy_vals[1].push_back(total_mean_field_term);
        energy_vals[2].push_back(total_cubic_term);
        energy_vals[3].push_back(total_entropy_term);
        energy_vals[4].push_back(total_L1_elastic_term);
        energy_vals[5].push_back(total_L2_elastic_term);
        energy_vals[6].push_back(total_L3_elastic_term);
    }
}

} // nematic_energy
