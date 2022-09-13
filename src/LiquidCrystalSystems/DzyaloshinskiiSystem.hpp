#ifndef DZYALOSHINSKII_SYSTEM_HPP
#define DZYALOSHINSKII_SYSTEM_HPP

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>


#include <string>

class DzyaloshinskiiSystem
{
public:
    DzyaloshinskiiSystem(double eps_, unsigned int degree);
    
    void make_grid(unsigned int n_refines);
    void setup_system();
    void assemble_system();
    void solve_and_update(double newton_step);
    double run_newton_method(double tol, 
                             unsigned int max_iter, 
                             double newton_step = 1.0);
    void output_solution(std::string filename);
    void output_hdf5(unsigned int n_points, std::string filename);

private:
    static constexpr int dim = 1;

    dealii::Triangulation<dim> tria;
    dealii::FE_Q<dim> fe;
    dealii::DoFHandler<dim> dof_handler;
    dealii::AffineConstraints<double> constraints;

    dealii::SparsityPattern sparsity_pattern;
    dealii::SparseMatrix<double> system_matrix;

    dealii::Vector<double> solution;
    dealii::Vector<double> system_rhs;

    double eps;
};

#endif
