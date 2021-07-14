#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template <int dim>
class UniformConfiguration : public Function<dim>
{
public:
	UniformConfiguration()
		: Function<dim>(dim)
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
	double return_value{ component == 0 ? 1.0 : 0.0 };
	return return_value;
}



template <int dim>
void UniformConfiguration<dim>::vector_value(const Point<dim> &p,
											 Vector<double> &value) const
{
	value[0] = 1.0;
	value[1] = 0.0;
	if (dim == 3) { value[2] = 0.0; }
}



template <int dim>
class PlusHalfDefect : public Function<dim>
{
public:
	PlusHalfDefect()
		: Function<dim>(dim)
	{}

	virtual double value(const Point<dim> &p,
						 const unsigned int component) const override;

	virtual void vector_value(const Point<dim> &p,
							  Vector<double> &value) const override;
};



template <int dim>
double PlusHalfDefect<dim>::value(const Point<dim> &p,
						   	      const unsigned int component) const
{
	double phi{std::atan2(p(1), p(0))};
	phi += 2.0*M_PI;
	phi = std::fmod(phi, 2.0*M_PI);
	double return_value;
	switch(component)
	{
	case 0:
		return_value = std::cos(phi / 2.0);
		break;
	case 1:
		return_value = std::sin(phi / 2.0);
		break;
	case 2:
		return_value = 0.0;
		break;
	}
	return return_value;
}



template <int dim>
void PlusHalfDefect<dim>::vector_value(const Point<dim> &p,
									   Vector<double> &value) const
{
	double phi{std::atan2(p(1), p(0))};
	phi += 2.0*M_PI;
	phi = std::fmod(phi, 2.0*M_PI);

	value[0] = std::cos(phi / 2.0);
	value[1] = std::sin(phi / 2.0);

	if (dim == 3) { value[2] = 0.0; }
}

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

};

template <int dim>
plot_uniaxial_nematic<dim>::plot_uniaxial_nematic()
	: dof_handler(triangulation)
	, fe(FE_Q<dim>(1), dim)
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
}

template <int dim>
void plot_uniaxial_nematic<dim>::output_results()
{
	std::vector<std::string> uniform_system_names(dim, "uniform_orientation");
	std::vector<std::string> defect_system_names(dim, "defect_orientation");
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
		data_component_interpretation(
				dim, DataComponentInterpretation::component_is_part_of_vector);

	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(uniform_system,
							 uniform_system_names,
							 DataOut<dim>::type_dof_data,
							 data_component_interpretation);
	data_out.add_data_vector(defect_system,
							 defect_system_names,
							 DataOut<dim>::type_dof_data,
							 data_component_interpretation);
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
	int refine_times{4};

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
