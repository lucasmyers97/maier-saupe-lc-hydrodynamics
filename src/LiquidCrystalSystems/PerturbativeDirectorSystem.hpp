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

template <int dim>
class PerturbativeDirectorRighthandSide : dealii::Function<dim>
{
public:
    PerturbativeDirectorRighthandSide(const std::vector<double> &defect_charges,
                                      const std::vector<dealii::Point<dim>> &defect_points,
                                      double eps)
        : dealii::Function<dim>()
        , defect_charges(defect_charges)
        , defect_points(defect_points)
        , eps(eps)
    {}

  virtual double value(const dealii::Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                          std::vector<double> &value_list,
                          const unsigned int component = 0) const override;

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
 
    PerturbativeDirectorSystem(unsigned int degree,
                               double left,
                               double right,
                               unsigned int num_refines,
                               unsigned int num_further_refines,
                               const std::vector<dealii::Point<dim>> &defect_pts,
                               const std::vector<double> &defect_refine_distances,
                               double defect_radius,
                               bool fix_defects,
                               BoundaryCondition boundary_condition,
                               std::unique_ptr<PerturbativeDirectorRighthandSide<dim>> righthand_side);

    void run();

private:
    // grid functions
    void make_grid();
    void refine_further();
    void refine_around_defects();

    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();

    // output functions
    void output_results(const unsigned int cycle) const;
    void output_points_to_hdf5() const;

    // grid parameters
    double left;
    double right;
    unsigned int num_refines;
    unsigned int num_further_refines;
    std::vector<dealii::Point<dim>> defect_pts;
    std::vector<double> defect_refine_distances;
    double defect_radius;
    bool fix_defects;

    BoundaryCondition boundary_condition;

    std::unique_ptr<PerturbativeDirectorRighthandSide<dim>> righthand_side;

    MPI_Comm mpi_communicator;

    dealii::Triangulation<dim> coarse_tria;
    dealii::parallel::distributed::Triangulation<dim> triangulation;

    dealii::FE_Q<dim>       fe;
    dealii::DoFHandler<dim> dof_handler;

    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;

    dealii::AffineConstraints<double> constraints;

    dealii::LinearAlgebraTrilinos::MPI::SparseMatrix system_matrix;
    dealii::LinearAlgebraTrilinos::MPI::Vector       locally_relevant_solution;
    dealii::LinearAlgebraTrilinos::MPI::Vector       system_rhs;

    dealii::ConditionalOStream pcout;
    dealii::TimerOutput        computing_timer;
};




#endif
