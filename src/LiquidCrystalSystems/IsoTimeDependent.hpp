#ifndef ISO_TIME_DEPENDENT_HPP
#define ISO_TIME_DEPENDENT_HPP

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

#include <deal.II/lac/vector.h>

#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/DefectConfiguration.hpp"
#include "BoundaryValues/UniformConfiguration.hpp"
#include "LagrangeMultiplier.hpp"

#include <memory>
#include <string>


template <int dim, int order>
class IsoTimeDependent
{
public:
    IsoTimeDependent();
    IsoTimeDependent(const boost::program_options::variables_map &vm);
    void run();
    void write_to_grid(const std::string grid_filename,
                       const std::string output_filename,
                       const std::vector<std::string> meshgrid_names,
                       double dist_scale) const;

  private:
    void make_grid(const unsigned int num_refines,
                   const double left = 0,
                   const double right = 1);
    void setup_system(bool initial_step);
    void assemble_system();
    void solve();
    void set_boundary_values();
    double determine_step_length();

    void output_results(const std::string data_folder,
                        const std::string filename) const;
    void save_data(const std::string data_folder,
                   const std::string filename) const;
    void output_grid(const std::string data_folder,
                     const std::string filename) const;
    void output_sparsity_pattern(const std::string data_folder,
                                 const std::string filename) const;

    friend class boost::serialization::access;

    template <class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        ar & triangulation;
        ar & dof_handler;
        ar & fe;
        ar & system_matrix;

        ar & hanging_node_constraints;
        // ar & boundary_value_func;

        ar & current_solution;
        ar & system_update;
        ar & system_rhs;

        ar & lagrange_multiplier;

        ar & left_endpoint;
        ar & right_endpoint;
        ar & num_refines;

        ar & simulation_step_size;
        ar & simulation_tol;
        ar & simulation_max_iters;
        ar & maier_saupe_alpha;
        ar & boundary_values_name;
        ar & S_value;
        ar & defect_charge_name;

        ar & data_folder;
        ar & initial_config_filename;
        ar & final_config_filename;
        ar & archive_filename;
    }

    template <class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        ar & triangulation;
        ar & dof_handler;
        ar & fe;
        ar & system_matrix;

        ar & hanging_node_constraints;

        // dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
        // dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
        // hanging_node_constraints.condense(dsp);
        // sparsity_pattern.copy_from(dsp);

        // ar & boundary_value_func;

        ar & current_solution;
        ar & system_update;
        ar & system_rhs;

        ar & lagrange_multiplier;

        ar & left_endpoint;
        ar & right_endpoint;
        ar & num_refines;

        ar & simulation_step_size;
        ar & simulation_tol;
        ar & simulation_max_iters;
        ar & maier_saupe_alpha;
        ar & boundary_values_name;
        ar & S_value;
        ar & defect_charge_name;

        ar & data_folder;
        ar & initial_config_filename;
        ar & final_config_filename;
        ar & archive_filename;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    dealii::Triangulation <dim> triangulation;
    dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe;
    dealii::SparsityPattern sparsity_pattern;
    dealii::SparseMatrix<double> system_matrix;

    dealii::AffineConstraints<double> hanging_node_constraints;
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    dealii::Vector<double> current_solution;
    dealii::Vector<double> system_update;
    dealii::Vector<double> system_rhs;

    LagrangeMultiplier<order> lagrange_multiplier;

    double left_endpoint;
    double right_endpoint;
    double num_refines;

    double simulation_step_size;
    double simulation_tol;
    int simulation_max_iters;
    double maier_saupe_alpha;

    std::string boundary_values_name;
    double S_value;
    std::string defect_charge_name;

    std::string data_folder;
    std::string initial_config_filename;
    std::string final_config_filename;
    std::string archive_filename;
};

#endif
