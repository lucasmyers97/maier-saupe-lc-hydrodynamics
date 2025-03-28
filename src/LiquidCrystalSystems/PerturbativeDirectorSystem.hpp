#ifndef PERTURBATIVE_DIRECTOR_SYSTEM_HPP
#define PERTURBATIVE_DIRECTOR_SYSTEM_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_pattern.h>

#include "Utilities/GridTools.hpp"

template <int dim>
class PerturbativeDirectorRighthandSide : public dealii::Function<dim>
{
public:
    PerturbativeDirectorRighthandSide(const std::vector<double> &defect_charges,
                                      const std::vector<dealii::Point<dim>> &defect_points)
        : dealii::Function<dim>()
        , defect_charges(defect_charges)
        , defect_points(defect_points)
    {}

  virtual double value(const dealii::Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                          std::vector<double> &value_list,
                          const unsigned int component = 0) const override;

private:
    std::vector<double> defect_charges;
    std::vector<dealii::Point<dim>> defect_points;
};



template <int dim>
class PerturbativeDirectorBoundaryCondition : public dealii::Function<dim>
{
public:
    PerturbativeDirectorBoundaryCondition(const std::vector<double> &defect_charges,
                                          const std::vector<dealii::Point<dim>> &defect_points,
                                          double eps)
        : dealii::Function<dim>(2)
        , defect_charges(defect_charges)
        , defect_points(defect_points)
        , eps(eps)
    {}

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;
    virtual void vector_value(const dealii::Point<dim> &p,
					          dealii::Vector<double> &value) const override;
    virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                            std::vector<double> &value_list,
                            const unsigned int component = 0) const override;
    virtual void
    vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                      std::vector<dealii::Vector<double>>   &value_list)
                      const override;

private:
    std::vector<double> defect_charges;
    std::vector<dealii::Point<dim>> defect_points;
    double eps;
};



template <int dim>
class PerturbativeDirectorSystem
{
public:
    enum class BoundaryCondition
    {
        Dirichlet,
        Neumann
    };

    enum class SolverType
    {
        Direct,
        CG
    };
 
    PerturbativeDirectorSystem(unsigned int degree,
                               std::string grid_name,
                               std::string grid_parameters,
                               double left,
                               double right,
                               unsigned int num_refines,
                               unsigned int num_further_refines,
                               const std::vector<dealii::Point<dim>> &defect_pts,
                               const std::vector<double> &defect_refine_distances,
                               double defect_radius,
                               bool fix_defects,
                               std::string grid_filename,

                               SolverType solver_type,

                               const std::string data_folder,
                               const std::string solution_vtu_filename,
                               const std::string rhs_vtu_filename,
                               const std::string outer_structure_filename,
                               const std::string dataset_name,
                               const std::string core_structure_filename,
                               const std::string pos_dataset_name,
                               const std::string neg_dataset_name,

                               const GridTools::RadialPointSet<dim> &point_set,
                               unsigned int refinement_level,
                               bool allow_merge,
                               unsigned int max_boxes,
                               BoundaryCondition boundary_condition,
                               std::unique_ptr<dealii::Function<dim>> righthand_side,
                               std::unique_ptr<dealii::Function<dim>> boundary_function);

    void run();

private:
    // grid functions
    void make_grid();
    void read_grid();
    void refine_further();
    void refine_around_defects();

    void setup_system();
    void setup_system_direct();
    void assemble_system();
    void assemble_system_direct();
    void solve();
    void solve_direct();
    void solve_mass_matrix();
    void solve_mass_matrix_direct();
    void refine_grid();

    // output functions
    void output_results(const unsigned int cycle) const;
    void output_rhs() const;
    void output_points_to_hdf5() const;
    void output_cores_to_hdf5() const;
    void output_archive() const;

    // grid parameters
    std::string grid_name;
    std::string grid_parameters;
    double left;
    double right;
    unsigned int num_refines;
    unsigned int num_further_refines;
    std::vector<dealii::Point<dim>> defect_pts;
    std::vector<double> defect_refine_distances;
    double defect_radius;
    bool fix_defects;
    std::string grid_filename;

    // solver type
    SolverType solver_type;

    // output parameters
    std::string data_folder;

    std::string solution_vtu_filename;
    std::string rhs_vtu_filename;

    std::string outer_structure_filename;
    std::string dataset_name;

    std::string core_structure_filename;
    std::string pos_dataset_name;
    std::string neg_dataset_name;

    GridTools::RadialPointSet<dim> point_set;
    unsigned int refinement_level = 3;
    bool allow_merge = false;
    unsigned int max_boxes = dealii::numbers::invalid_unsigned_int;

    // boundary stuff
    BoundaryCondition boundary_condition;
    std::unique_ptr<dealii::Function<dim>> righthand_side;
    std::unique_ptr<dealii::Function<dim>> boundary_function;

    MPI_Comm mpi_communicator;

    dealii::Triangulation<dim> coarse_tria;
    dealii::parallel::distributed::Triangulation<dim> triangulation;

    dealii::FE_Q<dim>       fe;
    dealii::DoFHandler<dim> dof_handler;

    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;

    dealii::AffineConstraints<double> constraints;

    dealii::LinearAlgebraTrilinos::MPI::SparseMatrix system_matrix;
    dealii::LinearAlgebraTrilinos::MPI::SparseMatrix mass_matrix;
    dealii::LinearAlgebraTrilinos::MPI::Vector       locally_relevant_solution;
    dealii::LinearAlgebraTrilinos::MPI::Vector       system_rhs;
    dealii::LinearAlgebraTrilinos::MPI::Vector       system_rhs_solution;

    dealii::SparsityPattern sparsity_pattern;
    dealii::SparseMatrix<double> system_matrix_direct;
    dealii::SparseMatrix<double> mass_matrix_direct;
    dealii::Vector<double>       locally_relevant_solution_direct;
    dealii::Vector<double>       system_rhs_direct;
    dealii::Vector<double>       system_rhs_solution_direct;

    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;
};




#endif
