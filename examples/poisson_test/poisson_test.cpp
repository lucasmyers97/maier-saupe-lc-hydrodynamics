#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

using namespace dealii;

template <int dim>
class PoissonSolver
{
public:
	PoissonSolver();
	void run();

private:
	void make_grid(double left = -1,
			       double right = 1,
				   int num_refines = 4);
	void output_grid();
	void setup_system();
	void assemble_system();
	void solve();
	void output_results();

	Triangulation<dim> tria;
	DoFHandler<dim> dof_handler;
	FE_Q<dim> fe;
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;

	AffineConstraints<double> hanging_node_constraints;

	Vector<double> solution;
	Vector<double> system_rhs;
};



template <int dim>
PoissonSolver<dim>::PoissonSolver()
	: dof_handler(tria)
	, fe(1)
{}



template <int dim>
void PoissonSolver<dim>::make_grid(double left, double right, int num_refines)
{
	GridGenerator::hyper_cube(tria, left, right);
	tria.refine_global(num_refines);
}



template <int dim>
void PoissonSolver<dim>::output_grid()
{
	std::ofstream out("grid_1.svg");

	GridOut grid_out;
	grid_out.write_svg(tria, out);
}



template <int dim>
void PoissonSolver<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	hanging_node_constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler,
											hanging_node_constraints);
	hanging_node_constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);

	hanging_node_constraints.condense(dsp);

	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit(sparsity_pattern);
}



template <int dim>
void PoissonSolver<dim>::assemble_system()
{
	QGauss<dim> quadrature_formula(fe.degree + 1);
	system_matrix = 0;
	system_rhs = 0;

	FEValues<dim> fe_values(fe,
							quadrature_formula,
							update_values
							| update_gradients
							| update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_values.reinit(cell);
		cell_matrix = 0;
		cell_rhs = 0;

		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) +=
						(-(fe_values.shape_grad(i, q)
						  * fe_values.shape_grad(j, q))
						 * fe_values.JxW(q));

				cell_rhs(i) +=
					fe_values.shape_value(i, q)
					* 1.0
					* fe_values.JxW(q);
			}
		}

		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j < dofs_per_cell; ++j)
				system_matrix.add(local_dof_indices[i],
								  local_dof_indices[j],
								  cell_matrix(i, j));

			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}

	hanging_node_constraints.condense(system_matrix);
	hanging_node_constraints.condense(system_rhs);

	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 Functions::ZeroFunction<dim>(),
											 boundary_values);
	MatrixTools::apply_boundary_values(boundary_values,
									   system_matrix,
									   solution,
									   system_rhs);
}



template <int dim>
void PoissonSolver<dim>::solve()
{
	SparseDirectUMFPACK solver;
	solver.factorize(system_matrix);
	solution = system_rhs;
	solver.solve(solution);
}



//template <int dim>
//void PoissonSolver<dim>::solve()
//{
//	SolverControl solver_control(1000, 1e-12);
//	SolverCG<Vector<double>> solver(solver_control);
//	solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
//}



template <int dim>
void PoissonSolver<dim>::output_results()
{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");

	data_out.build_patches();

	std::ofstream output(dim == 2 ? "solution-2d.vtu" : "solution-3d.vtu");
	data_out.write_vtu(output);
}



template <int dim>
void PoissonSolver<dim>::run()
{
	double left = -5;
	double right = 5;
	int num_refines = 7;

	make_grid(left, right, num_refines);
	output_grid();
	setup_system();
	assemble_system();
	solve();
	output_results();
}



int main()
{
	const int dim = 2;
	PoissonSolver<dim> poisson;
	poisson.run();

	return 0;
}
