#ifndef LIQUID_CRYSTAL_SYSTEM_HPP
#define LIQUID_CRYSTAL_SYSTEM_HPP

#include <boost/program_options.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/vector.h>

#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/DefectConfiguration.hpp"
#include "BoundaryValues/UniformConfiguration.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"

#include <memory>
#include <string>

template <int dim>
class LiquidCrystalSystem
{
public:
    LiquidCrystalSystem(const int order,
                        const dealii::Triangulation<dim> &triangulation);
    LiquidCrystalSystem(const int order,
                        const dealii::Triangulation<dim> &triangulation,
                        const unsigned int degree,
                        const std::string boundary_values_name,
                        const std::map<std::string, boost::any> &am,
                        const double lagrange_step_size,
                        const double lagrange_tol,
                        const unsigned int lagrange_max_iters,
                        const double maier_saupe_alpha_);

    void setup_system(bool initial_step);
    void assemble_system(double dt);
    void solve();
    void update_current_solution(double alpha);
    double return_norm();
    void set_past_solution_to_current();
    void output_results(const std::string data_folder,
                        const std::string filename,
                        const int timestep) const;

  private:
    /** \brief Connects mesh and FE functions to problem vector and matrix */
    dealii::DoFHandler<dim> dof_handler;
    /** \brief Takes care of values associated with FE basis functions */
    dealii::FESystem<dim> fe;
    /** \brief Deals with FE matrix sparsity pattern */
    dealii::SparsityPattern sparsity_pattern;
    /** \brief Matrix for the linear FE problem */
    dealii::SparseMatrix<double> system_matrix;

    /** \brief Takes care of assigning boundary values to FE vector */
    dealii::AffineConstraints<double> constraints;
    /** \brief Function which is evaluated at boundary to give Dirichlet vals */
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    /** \brief FE vector holding solution from previous timestep */
    dealii::Vector<double> past_solution;
    /** \brief FE vector holding current solution iteration */
    dealii::Vector<double> current_solution;
    /** \brief Update vector for Newton-Rhapson method */
    dealii::Vector<double> system_update;
    /** \brief FE system right-hand side for current iteration */
    dealii::Vector<double> system_rhs;

    /** \brief Object which handles Lagrange Multiplier inversion of Q-tensor */
    LagrangeMultiplierAnalytic<dim> lagrange_multiplier;

    /** \brief Alpha constant for bulk energy for the Maier-Saupe field theory*/
    double maier_saupe_alpha;
};

#endif
