#ifndef ISO_STEADY_STATE_HPP
#define ISO_STEADY_STATE_HPP

#include <boost/program_options.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/vector.h>

#include "BoundaryValues/BoundaryValues.hpp"
#include "LagrangeMultiplier.hpp"

#include <memory>
#include <string>


template <int dim, int order>
class IsoSteadyState
{
public:
    IsoSteadyState(const boost::program_options::variables_map &vm);
    void run();
	
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
