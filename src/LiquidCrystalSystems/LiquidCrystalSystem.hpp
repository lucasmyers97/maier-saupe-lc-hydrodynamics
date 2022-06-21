#ifndef LIQUID_CRYSTAL_SYSTEM_HPP
#define LIQUID_CRYSTAL_SYSTEM_HPP

#include <boost/program_options.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>

#include <deal.II/base/parameter_handler.h>
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

#include <deal.II/base/parameter_handler.h>

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
    LiquidCrystalSystem(const dealii::Triangulation<dim> &triangulation,
                        const unsigned int degree = 0,
                        const std::string boundary_values_name
                        = std::string("uniform"),
                        const std::map<std::string, boost::any> &am
                        = std::map<std::string, boost::any>(),
                        const double maier_saupe_alpha_ = 8.0,
                        const int order = 590,
                        const double lagrange_step_size = 1.0,
                        const double lagrange_tol = 1e-10,
                        const unsigned int lagrange_max_iters = 20);

    void declare_parameters(dealii::ParameterHandler &prm);
    void get_parameters(dealii::ParameterHandler &prm);

    void setup_system(bool initial_step);
    void assemble_system(double dt);
    void solve();
    void update_current_solution(double alpha);
    double return_norm();
    void set_past_solution_to_current();
    void output_results(const std::string data_folder,
                        const std::string filename,
                        const int timestep) const;

    const dealii::DoFHandler<dim>& return_dof_handler() const;
    const dealii::Vector<double> &return_current_solution() const;
    const double return_parameters() const;

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

    // This gives boost::serialization access to private members
    friend class boost::serialization::access;

    /**
     * \brief Serializes class so that it can be read back into a different file
     */
    template <class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        ar & fe;
        ar & dof_handler;
        ar & sparsity_pattern;

        ar & boundary_value_func;

        ar & past_solution;
        ar & current_solution;

        ar & lagrange_multiplier;

        ar & maier_saupe_alpha;
    }

    template <class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        ar & fe;
        dof_handler.distribute_dofs(fe);
        ar & dof_handler;
        ar &sparsity_pattern;

        ar & boundary_value_func;

        ar & past_solution;
        ar & current_solution;

        ar & lagrange_multiplier;

        ar & maier_saupe_alpha;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

#endif
