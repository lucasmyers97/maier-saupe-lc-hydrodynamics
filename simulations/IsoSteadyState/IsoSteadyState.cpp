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

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_postprocessor.h>

#include "LagrangeMultiplier.hpp"

#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

namespace {
	constexpr int order{590};
	double lagrange_alpha{1.0};
	double tol{1e-8};
	int max_iter{20};

	constexpr int Q_dim{5};
	constexpr int mat_dim{3};
	double alpha{8.0};
}

template <int dim>
class IsoSteadyState
{
public:
	IsoSteadyState();
	void run();
	
private:
	void make_grid(const unsigned int num_refines,
				   const double left = 0,
				   const double right = 1);
	void setup_system(bool initial_step);
	void assemble_system();
	void solve();
	void set_boundary_values();
	void output_results();

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



/*template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
			 	 	 	 	 	  const unsigned int component) const
{
	double phi{std::atan2(p(1), p(0))};
	phi += 2.0*M_PI;
	phi = std::fmod(phi, 2.0*M_PI);
	double return_value;
	double S{0.4};
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

	double S{0.4};

	value[0] = S * 0.5 * (3*std::cos(phi / 2.0)*std::cos(phi / 2.0) - 1.0);
	value[1] = S * 0.5 * 3*std::cos(phi / 2.0)*std::sin(phi / 2.0);
	value[2] = 0.0;
	value[3] = S * 0.5 * (3*std::sin(phi / 2.0)*std::sin(phi / 2.0) - 1.0);
	value[4] = 0.0;

}*/



template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
			 	 	 	 	 	  const unsigned int component) const
{
	double return_value;
	double S{0.6751};
	switch (component){
	case 0:
		return_value = S * 2.0/3.0;
		break;
	case 1:
		return_value = 0.0;
		break;
	case 2:
		return_value = 0.0;
		break;
	case 3:
		return_value = -S * 1.0/3.0;
		break;
	case 4:
		return_value = 0.0;
		break;
	}

	return return_value;
}



template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
									   Vector<double> &value) const
{
	double S{0.6751};
	value[0] = S * 2.0/3.0;
	value[1] = 0.0;
	value[2] = 0.0;
	value[3] = -S * 1.0 / 3.0;
	value[4] = 0.0;
}



template <int dim>
class DirectorPostprocessor : public DataPostprocessorVector<dim>
{
public:
	DirectorPostprocessor(std::string suffix)
		:
		DataPostprocessorVector<dim> ("n" + suffix, update_values)
	{}

	virtual void evaluate_vector_field
	(const DataPostprocessorInputs::Vector<dim> &input_data,
	 std::vector<Vector<double>> &computed_quantities) const override
	{
		AssertDimension(input_data.solution_values.size(),
					    computed_quantities.size());

		const double lower_bound = -5.0;
		const double upper_bound = 5.0;
		const double abs_accuracy = 1e-8;

		LAPACKFullMatrix<double> Q(mat_dim, mat_dim);
		FullMatrix<double> eigenvecs(mat_dim, mat_dim);
		Vector<double> eigenvals(mat_dim);
		for (unsigned int p=0; p<input_data.solution_values.size(); ++p)
		{
			AssertDimension(computed_quantities[p].size(), dim);

			Q.reinit(mat_dim, mat_dim);
			eigenvecs.reinit(mat_dim, mat_dim);
			eigenvals.reinit(mat_dim);

			// generate Q-tensor
			Q(0, 0) = input_data.solution_values[p][0];
			Q(0, 1) = input_data.solution_values[p][1];
			Q(0, 2) = input_data.solution_values[p][2];
			Q(1, 1) = input_data.solution_values[p][3];
			Q(1, 2) = input_data.solution_values[p][4];
			Q(1, 0) = Q(0, 1);
			Q(2, 0) = Q(0, 2);
			Q(2, 1) = Q(1, 2);
			Q(2, 2) = -(Q(0, 0) + Q(1, 1));

			Q.compute_eigenvalues_symmetric(lower_bound, upper_bound,
											abs_accuracy, eigenvals,
											eigenvecs);

			// Find index of maximal eigenvalue
			unsigned int max_entry{std::distance(eigenvals.begin(),
												 std::max_element(eigenvals.begin(),
														          eigenvals.end()))};
			computed_quantities[p][0] = eigenvecs(0, max_entry);
			computed_quantities[p][1] = eigenvecs(1, max_entry);
			if (dim == 3) { computed_quantities[p][2] = eigenvecs(2, max_entry); }
		}

	}

};



template <int dim>
IsoSteadyState<dim>::IsoSteadyState()
	: dof_handler(triangulation)
	, fe(FE_Q<dim>(1), Q_dim)
{}


template <int dim>
void IsoSteadyState<dim>::make_grid(const unsigned int num_refines,
									const double left,
									const double right)
{
	GridGenerator::hyper_cube(triangulation, left, right);
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

		VectorTools::project(dof_handler,
							 hanging_node_constraints,
							 QGauss<dim>(fe.degree + 1),
							 BoundaryValues<dim>(),
							 current_solution);
	}
	system_update.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);

	hanging_node_constraints.condense(dsp);

	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit(sparsity_pattern);
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
						  * fe_values.shape_grad(j, q))
						 -
						 (fe_values.shape_value(i, q)
						  * R_inv_phi[j][component_i]))
						 * fe_values.JxW(q);
				}
				cell_rhs(i) +=
					(-(fe_values.shape_value(i, q)
					   * alpha
					   * old_solution_values[q][component_i])
					  +
					  (fe_values.shape_grad(i, q)
					   * old_solution_gradients[q][component_i])
					  +
					  (fe_values.shape_value(i, q)
					   * Lambda[component_i]))
					  * fe_values.JxW(q);
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
	SparseDirectUMFPACK solver;
	solver.factorize(system_matrix);
	system_update = system_rhs;
	solver.solve(system_update);
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
void IsoSteadyState<dim>::output_results()
{
	DirectorPostprocessor<dim> director_postprocessor_defect("defect");
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(current_solution, director_postprocessor_defect);
	data_out.build_patches();

	std::cout << "Outputting results" << std::endl;

	std::ofstream output(
			"system-" + Utilities::int_to_string(dim) + "d.vtu");
	data_out.write_vtu(output);
}




template <int dim>
void IsoSteadyState<dim>::run()
{
	int num_refines = 4;
	double left = -1;
	double right = 1;
	make_grid(num_refines, left, right);

	std::string grid_filename = "grid_1.svg";
	// output_grid(grid_filename);

	setup_system(true);
	set_boundary_values();
	output_results();

	assemble_system();
	std::cout << "System assembled" << std::endl;
	solve();

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

