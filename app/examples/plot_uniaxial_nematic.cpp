#include <math.h>
#include <algorithm>

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include "BoundaryValues/DefectConfiguration.hpp"

using namespace dealii;

namespace {
	unsigned int vec_dim = 5;
	unsigned int mat_dim = 3;
}

template <int dim>
class UniformConfiguration : public Function<dim>
{
public:
	UniformConfiguration()
		: Function<dim>(vec_dim)
	{}

	virtual double value(const Point<dim> &p,
                       const unsigned int component) const override;
	virtual void vector_value(const Point<dim> &p,
                            Vector<double> &value) const override;
};



template <int dim>
double UniformConfiguration<dim>::value(const Point<dim> &p,
                                        const unsigned int component) const
{
    double return_value = static_cast<double>(component == 0 ? 1 : 0);
	return return_value;
}



template <int dim>
void UniformConfiguration<dim>::vector_value(const Point<dim> &p,
                                             Vector<double> &value) const
{
	value[0] = 1.0;
	value[1] = 0.0;
	value[2] = 0.0;
	value[3] = 0.0;
	value[4] = 0.0;
}



template <int dim>
class PlusHalfDefect : public Function<dim>
{
public:
	PlusHalfDefect()
		: Function<dim>(vec_dim)
	{}

	virtual double value(const Point<dim> &p,
                       const unsigned int component) const override;
  virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                          std::vector<double> &value_list,
                          const unsigned int component = 0) const override;

	virtual void vector_value(const Point<dim> &p,
                            Vector<double> &value) const override;
};



template <int dim>
double PlusHalfDefect<dim>::value(const Point<dim> &p,
                                  const unsigned int component) const
{
	double phi{std::atan2(p(1), p(0))};
	double return_value;
	switch(component)
	{
	case 0:
		return_value = 0.5  * ( 1.0/3.0 + std::cos(2*0.5*phi) );
		break;
	case 1:
		return_value = 0.5  * std::sin(2*0.5*phi);
		break;
	case 2:
		return_value = 0.0;
		break;
	case 3:
		return_value = 0.5  * ( 1.0/3.0 - std::cos(2*0.5*phi) );
		break;
	case 4:
		return_value = 0.0;
		break;
	}
	return return_value;
}




template <int dim>
void PlusHalfDefect<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
	Assert(point_list.size() == value_list.size(), "size does not match");

	double k = 0.5;
	double S = 1.0;
	std::vector<double> phi_list(point_list.size());
	for (int i = 0; i < point_list.size(); ++i)
		phi_list[i] = std::atan2(point_list[i][1], point_list[i][0]);

	switch (component){
	case 0:
        for (int i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.5 * S * ( 1.0/3.0 + std::cos(2*k*phi_list[i]) );
		break;
	case 1:
        for (int i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.5 * S * std::sin(2*k*phi_list[i]);
		break;
	case 2:
		std::cout << "got here" << std::endl;
        for (int i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.0;
		break;
	case 3:
        for (int i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.5 * S * ( 1.0/3.0 - std::cos(2*k*phi_list[i]) );
		break;
	case 4:
        for (int i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.0;
		break;
	}
}
template <int dim>
void PlusHalfDefect<dim>::vector_value(const Point<dim> &p,
                                       Vector<double> &value) const
{
	double phi{std::atan2(p(1), p(0))};

	value[0] = 0.5  * ( 1.0/3.0 + std::cos(2*0.5*phi) );
	value[1] = 0.5  * std::sin(2*0.5*phi);
	value[2] = 0.0;
	value[3] = 0.5  * ( 1.0/3.0 - std::cos(2*0.5*phi) );
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

      // iterator of maximal eigenvalue
      auto max_iter = std::max_element(eigenvals.begin(), eigenvals.end());
			// Find index of maximal eigenvalue
			auto max_idx = std::distance(eigenvals.begin(), max_iter);

			computed_quantities[p][0] = eigenvecs(0, max_idx);
			computed_quantities[p][1] = eigenvecs(1, max_idx);
			if (dim == 3) { computed_quantities[p][2] = eigenvecs(2, max_idx); }
		}

	}

};



template <int dim>
class plot_uniaxial_nematic
{
public:
	plot_uniaxial_nematic();
	void run();

private:
	void generate_grid(double left=0, double right=1, int times=1);
	void setup_system();
	void project_system();
	void output_results();
	void output_grid();

	Triangulation<dim> triangulation;
	DoFHandler<dim> dof_handler;

	FESystem<dim> fe;

	AffineConstraints<double> constraints;

	Vector<double> uniform_system;
	Vector<double> defect_system;
	Vector<double> external_defect_system;

};

template <int dim>
plot_uniaxial_nematic<dim>::plot_uniaxial_nematic()
	: dof_handler(triangulation)
	, fe(FE_Q<dim>(1), vec_dim)
{}

template <int dim>
void plot_uniaxial_nematic<dim>::generate_grid(double left,
											  double right,
											  int times)
{
	GridGenerator::hyper_cube(triangulation, left, right);
	triangulation.refine_global(times);
}



template <int dim>
void plot_uniaxial_nematic<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);
	uniform_system.reinit(dof_handler.n_dofs());
	defect_system.reinit(dof_handler.n_dofs());
	external_defect_system.reinit(dof_handler.n_dofs());

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();
}



template <int dim>
void plot_uniaxial_nematic<dim>::project_system()
{
	VectorTools::project(dof_handler,
						 constraints,
						 QGauss<dim>(fe.degree + 1),
						 UniformConfiguration<dim>(),
						 uniform_system);

	VectorTools::project(dof_handler,
						 constraints,
						 QGauss<dim>(fe.degree + 1),
						 PlusHalfDefect<dim>(),
						 defect_system);

	VectorTools::project(dof_handler,
						 constraints,
						 QGauss<dim>(fe.degree + 1),
						 DefectConfiguration<dim>(),
						 external_defect_system);
}

template <int dim>
void plot_uniaxial_nematic<dim>::output_results()
{
	DirectorPostprocessor<dim> director_postprocessor_uniform("uniform");
	DirectorPostprocessor<dim> director_postprocessor_defect("defect");
	DirectorPostprocessor<dim> 
		director_postprocessor_ext_defect("external_defect");
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(uniform_system, "uniform_Q");
	data_out.add_data_vector(defect_system, "defect_Q");
	data_out.add_data_vector(external_defect_system, "ext_defect_Q");

	data_out.add_data_vector(uniform_system, director_postprocessor_uniform);
	data_out.add_data_vector(defect_system, director_postprocessor_defect);
	data_out.add_data_vector(external_defect_system,
							 director_postprocessor_ext_defect);
	data_out.build_patches();

	std::ofstream output(
			"system-" + Utilities::int_to_string(dim) + "d.vtu");
	data_out.write_vtu(output);

}

template <int dim>
void plot_uniaxial_nematic<dim>::output_grid()
{
	std::ofstream out("grid-1.svg");
	GridOut grid_out;
	grid_out.write_svg(triangulation, out);
	std::cout << "Grid written to grid-1.svg" << std::endl;
}

template <int dim>
void plot_uniaxial_nematic<dim>::run()
{
	double left_endpoint{-1.0};
	double right_endpoint{1.0};
	int refine_times{6};

	generate_grid(left_endpoint,
				  right_endpoint,
				  refine_times);
	output_grid();

	setup_system();
	project_system();
	output_results();
}

int main()
{
	const int dim{2};
	plot_uniaxial_nematic<dim> pud;

	pud.run();

	return 0;
}
