#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataSpace.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

template <int dim>
class DefectToGrid
{
using mat =  std::vector<std::vector<double>>;

public:
    DefectToGrid();
    void run();

private:
    static constexpr int vec_dim = 5;

    void read_grid(std::string filename);
    void read_fe_system(std::string filename);
    void read_fe_at_gridpoints();
    void output_fe_vals(std::string filename);

    mat X;
    mat Y;
    int n;
    int m;
    int n_pts;

    dealii::Triangulation <dim> triangulation;
    dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe;
    dealii::Vector<double> solution;

    std::vector<mat> Q_vec = std::vector<mat>(vec_dim);
};



template <int dim>
DefectToGrid<dim>::DefectToGrid()
	: dof_handler(triangulation)
	, fe(dealii::FE_Q<dim>(1), DefectToGrid<dim>::vec_dim)
{}



template <int dim>
void DefectToGrid<dim>::read_grid(std::string filename)
{
    // read file
    HighFive::File f(filename);

    // get object info and create dataset vector
    size_t num_objects = f.getNumberObjects();
    std::vector<std::string> object_names = f.listObjectNames();
    std::vector<HighFive::DataSet> dset(num_objects);

    // just read out grid position data
    dset[num_objects - 2] = f.getDataSet(object_names[num_objects - 2]);
    dset[num_objects - 2].read(X);
    dset[num_objects - 1] = f.getDataSet(object_names[num_objects - 1]);
    dset[num_objects - 1].read(Y);

    // get number of points
    m = X.size();
    n = X[0].size();
    n_pts = m*n;
}



template <int dim>
void DefectToGrid<dim>::read_fe_system(std::string filename)
{
    // read from archive file
    std::ifstream ifs(filename);
	boost::archive::text_iarchive ia(ifs);

    // read into solution and dof_handler
	solution.load(ia, 1);
    
    dealii::GridGenerator::hyper_cube(triangulation, -10/sqrt(2), 10/sqrt(2));
	triangulation.refine_global(8);
    dof_handler.distribute_dofs(fe);
	// dof_handler.load(ia, 1);
}



template <int dim>
void DefectToGrid<dim>::read_fe_at_gridpoints()
{
    std::vector<dealii::Point<dim>> points(n_pts);
    std::vector<dealii::Vector<double>> values
        (n_pts, dealii::Vector<double> (vec_dim));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            points[i*n + j](0) = X[i][j] / sqrt(2);
            points[i*n + j](1) = Y[i][j] / sqrt(2);
        }
    }

    dealii::Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
    fe_function.vector_value_list(points, values);

    // put values from fe solution into Q_vec object
    for (int vec_idx = 0; vec_idx < vec_dim; ++vec_idx)
    {
        // resize to be shape of grid
        Q_vec[vec_idx].resize(m);
        for (int i = 0; i < m; ++i)
        {
            Q_vec[vec_idx][i].resize(n);
        }

        // populate with values
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                Q_vec[vec_idx][i][j] = values[i*n + j][vec_idx];
                    // std::cout << Q_vec[vec_idx][i][j] << std::endl;
            }
        }
    }
}



template <int dim>
void DefectToGrid<dim>::output_fe_vals(std::string filename)
{
    HighFive::File f(filename, 
                     HighFive::File::ReadWrite | 
                     HighFive::File::Create | 
                     HighFive::File::Truncate);
    std::vector<HighFive::DataSet> dset(vec_dim + dim);

    dset[0] = f.createDataSet<double>("Q1", HighFive::DataSpace::From(Q_vec[0]));
    dset[1] = f.createDataSet<double>("Q2", HighFive::DataSpace::From(Q_vec[1]));
    dset[2] = f.createDataSet<double>("Q3", HighFive::DataSpace::From(Q_vec[2]));
    dset[3] = f.createDataSet<double>("Q4", HighFive::DataSpace::From(Q_vec[3]));
    dset[4] = f.createDataSet<double>("Q5", HighFive::DataSpace::From(Q_vec[4]));
    dset[5] = f.createDataSet<double>("X", HighFive::DataSpace::From(X));
    dset[6] = f.createDataSet<double>("Y", HighFive::DataSpace::From(Y));

    for (int i = 0; i < vec_dim; ++i)
        dset[i].write(Q_vec[i]);

    dset[5].write(X);
    dset[6].write(Y);
}



template <int dim>
void DefectToGrid<dim>::run()
{
    std::string grid_filename = "/home/lucasmyers97/"
                                "maier-saupe-lc-hydrodynamics/examples/"
                                "cody_data/plus-half-defect-cody.h5"; 
    read_grid(grid_filename);

    std::string fe_filename = "/home/lucasmyers97/"
                              "maier-saupe-lc-hydrodynamics/data/"
                              "IsoSteadyState/2021-10-01/"
                              "save-data-plus-half-8.dat";
    read_fe_system(fe_filename);
    read_fe_at_gridpoints();

    std::string fe_output_filename = "/home/lucasmyers97/"
                                     "maier-saupe-lc-hydrodynamics/data/"
                                     "IsoSteadyState/2021-10-01/"
                                     "plus-half-defect-me.hdf5";
    output_fe_vals(fe_output_filename);
}



int main()
{
    const int dim = 2;
    DefectToGrid<dim> defect_to_grid;
    defect_to_grid.run();

    return 0;
}