#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

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
	Triangulation <dim> triangulation;
	void make_grid(const unsigned int num_refines);
	void output_grid(const char* filename);
};



template <int dim>
IsoSteadyState<dim>::IsoSteadyState()
{}


template <int dim>
void IsoSteadyState<dim>::make_grid(const unsigned int num_refines)
{
	GridGenerator::hyper_cube(triangulation);
	triangulation.refine_global(num_refines);
}


template <int dim>
void IsoSteadyState<dim>::output_grid(const char* filename)
{
	std::ofstream out(filename);
	GridOut grid_out;
	grid_out.write_svg(triangulation, out);
	std::cout << "Grid written to " << filename << std::endl;
}


template <int dim>
void IsoSteadyState<dim>::run()
{
	int num_refines = 4;
	make_grid(num_refines);

	char* filename = "grid_1.svg";
	output_grid(filename);
}

int main()
{
	const int dim = 2;
	IsoSteadyState<dim> iso_steady_state;
	iso_steady_state.run();

	return 0;
}

