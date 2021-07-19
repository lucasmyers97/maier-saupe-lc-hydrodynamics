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
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/array_view.h>

#include <deal.II/dofs/dof_renumbering.h>

#include "LagrangeMultiplier.hpp"

#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

namespace {
	constexpr int order{110};
	double lagrange_alpha{1.0};
	double tol{1e-8};
	int max_iter{20};

	constexpr int Q_dim{5};
	double alpha{1.0};
}

template <int dim>
class IsoSteadyState
{
public:
	IsoSteadyState();
	void run();
	
private:
	void make_grid(const unsigned int num_refines);
	void setup_system(bool initial_step);
	void assemble_system();
	void solve();
	void set_boundary_values();

	Triangulation <dim> triangulation;
	DoFHandler<dim> dof_handler;
	FESystem<dim> fe;
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;

	AffineConstraints<double> hanging_node_constraints;

	Vector<double> current_solution;
	Vector<double> system_update;
	Vector<double> system_rhs;

	void output_grid(const std::string filename);
	void output_sparsity_pattern(const std::string filename);
};



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
	BoundaryValues()
		: Function<dim>(Q_dim)
	{}

	virtual double value(const Point<dim> &p,
					     const unsigned int component = 0) const override;

	virtual void vector_value(const Point<dim> &p,
						      Vector<double> &value) const override;
};



template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
			 	 	 	 	 	  const unsigned int component) const
{
	double phi{std::atan2(p(1), p(0))};
	phi += 2.0*M_PI;
	phi = std::fmod(phi, 2.0*M_PI);
	double return_value;
	double S{0.6};
	switch(component)
	{
	case 0:
		return_value = S * 0.5
					   * (3*std::cos(phi / 2.0)*std::cos(phi / 2.0) - 1.0);
		break;
	case 1:
		return_value = S * 0.5 * 3*std::cos(phi / 2.0)*std::sin(phi / 2.0);
		break;
	case 2:
		return_value = 0.0;
		break;
	case 3:
		return_value = S * 0.5
		               * (3*std::sin(phi / 2.0)*std::sin(phi / 2.0) - 1.0);
		break;
	case 4:
		return_value = 0.0;
		break;
	}
	return S*return_value;
}



template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
									   Vector<double> &value) const
{
	double phi{std::atan2(p(1), p(0))};
	phi += 2.0*M_PI;
	phi = std::fmod(phi, 2.0*M_PI);

	double S{0.6};

	value[0] = S * 0.5 * (3*std::cos(phi / 2.0)*std::cos(phi / 2.0) - 1.0);
	value[1] = S * 0.5 * 3*std::cos(phi / 2.0)*std::sin(phi / 2.0);
	value[2] = 0.0;
	value[3] = S * 0.5 * (3*std::sin(phi / 2.0)*std::sin(phi / 2.0) - 1.0);
	value[4] = 0.0;

}



template <int dim>
IsoSteadyState<dim>::IsoSteadyState()
	: dof_handler(triangulation)
	, fe(FE_Q<dim>(1), Q_dim)
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
void IsoSteadyState<dim>::setup_system(bool initial_step)
{
	if (initial_step) {
		dof_handler.distribute_dofs(fe);
		current_solution.reinit(dof_handler.n_dofs());

		hanging_node_constraints.clear();
		DoFTools::make_hanging_node_constraints(dof_handler,
												hanging_node_constraints);
		hanging_node_constraints.close();
	}
	system_update.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);

	hanging_node_constraints.condense(dsp);

	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit(sparsity_pattern);


	system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void IsoSteadyState<dim>::assemble_system()
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

	std::vector<std::vector<Tensor<1, dim>>>
	old_solution_gradients(n_q_points,
						   std::vector<Tensor<1, dim, double>>(fe.components));
	std::vector<Vector<double>>
	old_solution_values(n_q_points, Vector<double>(fe.components));
	Vector<double> Lambda(fe.components);
	LAPACKFullMatrix<double> R(fe.components, fe.components);
	std::vector<Vector<double>> R_inv_phi(dofs_per_cell,
										  Vector<double>(fe.components));

	LagrangeMultiplier<order> lagrange_multiplier(lagrange_alpha,
												  tol, max_iter);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		cell_matrix = 0;
		cell_rhs = 0;

		fe_values.reinit(cell);
		fe_values.get_function_gradients(current_solution,
										 old_solution_gradients);
		fe_values.get_function_values(current_solution,
									  old_solution_values);

		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			Lambda.reinit(fe.components);
			R.reinit(fe.components);

			lagrange_multiplier.setQ(old_solution_values[q]);
			lagrange_multiplier.returnLambda(Lambda);
			lagrange_multiplier.returnJac(R);
			R.compute_lu_factorization();

			for (unsigned int j = 0; j < dofs_per_cell; ++j)
			{
				const unsigned int component_j =
						fe.system_to_component_index(j).first;
				R_inv_phi[j].reinit(fe.components);
				R_inv_phi[j][component_j] = fe_values.shape_value(j, q);
				R.solve(R_inv_phi[j]);
			}

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				const unsigned int component_i =
					fe.system_to_component_index(i).first;

				for (unsigned int j = 0; j < dofs_per_cell; ++j)
				{
					const unsigned int component_j =
						fe.system_to_component_index(j).first;

					cell_matrix(i, j) +=
						((fe_values.shape_value(i, q)
						  * alpha
						  * fe_values.shape_value(j, q))
						 -
						 (fe_values.shape_grad(i, q)
						  *fe_values.shape_grad(j, q))
						 -
						 (fe_values.shape_value(i, q)
						  *R_inv_phi[j][component_j]))*fe_values.JxW(q);
				}
				cell_rhs(i) +=
					-((fe_values.shape_value(i, q)
					   * alpha
					   * old_solution_values[q][component_i])
					  +
					  (fe_values.shape_grad(i, q)
					   * old_solution_gradients[q][component_i])
					  +
					  (fe_values.shape_value(i, q)
					   * Lambda[component_i]))*fe_values.JxW(q);
			}
		}

		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j < dofs_per_cell; ++j)
			{
				system_matrix.add(local_dof_indices[i],
								  local_dof_indices[j],
								  cell_matrix(i, j));
			}
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}

	hanging_node_constraints.condense(system_matrix);
	hanging_node_constraints.condense(system_rhs);

	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 Functions::ZeroFunction<dim>(Q_dim),
											 boundary_values);
	MatrixTools::apply_boundary_values(boundary_values,
									   system_matrix,
									   system_update,
									   system_rhs);
}



template <int dim>
void IsoSteadyState<dim>::solve()
{

}



template <int dim>
void IsoSteadyState<dim>::set_boundary_values()
{
	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 BoundaryValues<dim>(),
											 boundary_values);
	for (auto &boundary_value : boundary_values)
		current_solution(boundary_value.first) = boundary_value.second;

	hanging_node_constraints.distribute(current_solution);
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

	setup_system(true);
	set_boundary_values();
	assemble_system();

	std::cout << system_rhs << std::endl;

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

