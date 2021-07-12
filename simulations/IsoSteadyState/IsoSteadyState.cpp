#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template <int dim>
class IsoSteadyState
{
public:
	IsoSteadyState();
	void run();
	
private:
	void make_grid(const unsigned int num_refines);
	void setup_system();
	void assemble_system();

	Triangulation <dim> triangulation;
	DoFHandler<dim> dof_handler;
	FESystem<dim> fe;
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;

	Vector<double> solution;
	Vector<double> system_rhs;

	void output_grid(const std::string filename);
	void output_sparsity_pattern(const std::string filename);
};



template <int dim>
IsoSteadyState<dim>::IsoSteadyState()
	: dof_handler(triangulation)
	, fe(FE_Q<dim>(1), dim*(dim - 1)/2 + (dim - 1))
{}


template <int dim>
void IsoSteadyState<dim>::make_grid(const unsigned int num_refines)
{
	GridGenerator::hyper_cube(triangulation);
	triangulation.refine_global(num_refines);
}


template <int dim>
void IsoSteadyState<dim>::output_grid(const std::string filename)
{
	std::ofstream out(filename);
	GridOut grid_out;
	grid_out.write_svg(triangulation, out);
	std::cout << "Grid written to " << filename << std::endl;
}



template <int dim>
void IsoSteadyState<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit(sparsity_pattern);

	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void IsoSteadyState<dim>::assemble_system()
{
	QGauss<dim> quadrature_formula(fe.degree + 1);
	FEValues<dim> fe_values(fe,
							quadrature_formula,
							update_values
							| update_gradients
							| update_JxW_values);
	const unsigned int dofs_per_cell = fe.dofs_per_cell();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);



}


template <int dim>
void IsoSteadyState<dim>::output_sparsity_pattern(const std::string filename)
{
	std::ofstream out(filename);
	sparsity_pattern.print_svg(out);
}




template <int dim>
void IsoSteadyState<dim>::run()
{
	int num_refines = 4;
	make_grid(num_refines);

	std::string grid_filename = "grid_1.svg";
	output_grid(grid_filename);

	setup_system();

	std::string sparsity_filename = "sparsity_pattern_1.svg";
	output_sparsity_pattern(sparsity_filename);
}



int main()
{
	const int dim = 2;
	IsoSteadyState<dim> iso_steady_state;
	iso_steady_state.run();

	return 0;
}

