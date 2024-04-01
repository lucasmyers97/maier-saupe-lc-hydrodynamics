#ifndef NEMATIC_ENERGY_HPP
#define NEMATIC_ENERGY_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/dofs/dof_handler.h>

#include "Numerics/LagrangeMultiplierAnalytic.hpp"

namespace nematic_energy
{

namespace LA = dealii::LinearAlgebraTrilinos;

template <int dim>
void singular_potential_energy(const MPI_Comm &mpi_communicator, 
                               double current_time,
                               double alpha, double L2, double L3,
                               const dealii::DoFHandler<dim> &dof_handler,
                               const LA::MPI::Vector &current_solution,
                               LagrangeMultiplierAnalytic<dim> &singular_potential,
                               std::vector<std::vector<double>> &energy_vals);

template <int dim>
void singular_potential_rot_energy(const MPI_Comm &mpi_communicator, 
                                   double current_time,
                                   double alpha, double L2, double L3, double omega,
                                   const dealii::DoFHandler<dim> &dof_handler,
                                   const LA::MPI::Vector &current_solution,
                                   LagrangeMultiplierAnalytic<dim> &singular_potential,
                                   std::vector<std::vector<double>> &energy_vals);

} // nematic_energy

#endif 
