#ifndef NEMATIC_ASSEMBLY_HPP
#define NEMATIC_ASSEMBLY_HPP

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/dofs/dof_handler.h>

#include "Numerics/LagrangeMultiplierAnalytic.hpp"

namespace nematic_assembly
{

namespace LA = dealii::LinearAlgebraTrilinos;

template <int dim>
void singular_potential_semi_implicit(double dt, double theta, double alpha,
                                      double L2, double L3,
                                      const dealii::DoFHandler<dim> &dof_handler,
                                      const LA::MPI::Vector &current_solution,
                                      const LA::MPI::Vector &past_solution,
                                      LagrangeMultiplierAnalytic<dim> singular_potential,
                                      const dealii::AffineConstraints<double> &constraints,
                                      LA::MPI::SparseMatrix &system_matrix,
                                      LA::MPI::Vector &system_rhs);

template <int dim>
void singular_potential_convex_splitting(double dt, double alpha,
                                         double L2, double L3,
                                         const dealii::DoFHandler<dim> &dof_handler,
                                         const LA::MPI::Vector &current_solution,
                                         const LA::MPI::Vector &past_solution,
                                         LagrangeMultiplierAnalytic<dim> singular_potential,
                                         const dealii::AffineConstraints<double> &constraints,
                                         LA::MPI::SparseMatrix &system_matrix,
                                         LA::MPI::Vector &system_rhs);

template <int dim>
void landau_de_gennes_convex_splitting(double dt, double A, double B, double C,
                                       double L2, double L3,
                                       const dealii::DoFHandler<dim> &dof_handler,
                                       const LA::MPI::Vector &current_solution,
                                       const LA::MPI::Vector &past_solution,
                                       const dealii::AffineConstraints<double> &constraints,
                                       LA::MPI::SparseMatrix &system_matrix,
                                       LA::MPI::Vector &system_rhs);

template <int dim>
void singular_potential_newtons_method(double alpha, double L2, double L3,
                                       const dealii::DoFHandler<dim> &dof_handler,
                                       const LA::MPI::Vector &current_solution,
                                       const LA::MPI::Vector &past_solution,
                                       LagrangeMultiplierAnalytic<dim> singular_potential,
                                       const dealii::AffineConstraints<double> &constraints,
                                       LA::MPI::SparseMatrix &system_matrix,
                                       LA::MPI::Vector &system_rhs);

} // nematic_assembly

#endif
