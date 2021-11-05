#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>

#include <vector>
#include <string>
#include <fstream>

#include "LinearInterpolation.hpp"
#include "LiquidCrystalPostprocessor.hpp"



template <int dim>
class DataToFe
{
public:
    DataToFe();
    void run();

private:
    void make_grid(const unsigned int num_refines,
                   const double left = 0,
                   const double right = 1);
	void setup_system();
    void read_data(const std::string filename);
    void output_results();

    dealii::Triangulation <dim> triangulation;
	dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe;
    dealii::AffineConstraints<double> hanging_node_constraints;

    dealii::Vector<double> solution_vec;

    LinearInterpolation<dim> sol;

    static constexpr int vec_dim = 5;
};



template <int dim>
DataToFe<dim>::DataToFe()
	: dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(1), vec_dim)
{}



template <int dim>
void DataToFe<dim>::make_grid(const unsigned int num_refines,
							  const double left,
							  const double right)
{
	dealii::GridGenerator::hyper_cube(triangulation, left, right);
	triangulation.refine_global(num_refines);
}



template <int dim>
void DataToFe<dim>::read_data(const std::string filename)
{
    HighFive::File f(filename);
    size_t num_objects = f.getNumberObjects();
    std::vector<std::string> object_names = f.listObjectNames();

    using mat =  std::vector<std::vector<double>>;
    std::vector<mat> Q_vec(vec_dim);
    mat X;
    mat Y;

    std::vector<HighFive::DataSet> dset(num_objects);

    for (int i = 0; i < vec_dim; ++i)
    {
        dset[i] = f.getDataSet(object_names[i]);
        dset[i].read(Q_vec[i]);
    }

    dset[num_objects - 2] = f.getDataSet(object_names[num_objects - 2]);
    dset[num_objects - 2].read(X);
    dset[num_objects - 1] = f.getDataSet(object_names[num_objects - 1]);
    dset[num_objects - 1].read(Y);

    sol.reinit(Q_vec, X, Y);
}



template <int dim>
void DataToFe<dim>::setup_system()
{

    dof_handler.distribute_dofs(fe);
    solution_vec.reinit(dof_handler.n_dofs());

    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                    hanging_node_constraints);
    hanging_node_constraints.close();

    dealii::VectorTools::project
        (dof_handler, hanging_node_constraints, 
         dealii::QGauss<dim>(fe.degree + 1), sol, solution_vec);

    // dealii::VectorTools::interpolate(dof_handler, sol, solution_vec);
}



template <int dim>
void DataToFe<dim>::output_results()
{
	DirectorPostprocessor<dim> director_postprocessor_defect("defect");
	SValuePostprocessor<dim> S_value_postprocessor_defect("defect");
	dealii::DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution_vec, director_postprocessor_defect);
	data_out.add_data_vector(solution_vec, S_value_postprocessor_defect);
	data_out.build_patches();

	std::cout << "Outputting results" << std::endl;

	std::string filename = "system-" + dealii::Utilities::int_to_string(dim) 
                           + "d" + "-plus-half.vtu";

	std::ofstream output(filename);
	data_out.write_vtu(output);
}



template <int dim>
void DataToFe<dim>::run()
{
    std::string filename = DATA_FILE_LOCATION;

	int num_refines{8};
	double left{-9.9};
	double right{9.9};
    make_grid(num_refines, left, right);

    read_data(filename);
    setup_system();

    // int num_vals = 1;
    // std::vector<dealii::Vector<double>> 
    //     vals(num_vals, dealii::Vector<double>(vec_dim));
    // std::vector<dealii::Point<dim>> p(num_vals, dealii::Point<dim>(5.5, 3.14));
    // sol.vector_value_list(p, vals);
    // std::cout << vals[0] << std::endl;

    output_results();
}



int main()
{
    const int dim = 2;
	DataToFe<dim> data_to_fe;
	data_to_fe.run();

    return 0;
}