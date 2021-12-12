#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <highfive/H5Easy.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/data_out.h>

#include <boost/program_options.hpp>

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Utilities/LinearInterpolation.hpp"
#include "Postprocessors/DirectorPostprocessor.hpp"
#include "Postprocessors/SValuePostprocessor.hpp"

namespace{
    constexpr int vec_dim = 5;
    constexpr int dim = 2;
    using namespace dealii;
    using mat = std::vector<std::vector<double>>;
    namespace po = boost::program_options;
}


int main(int ac, char* av[])
{
    // parse command-line options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("data-filename", po::value<std::string>(),
         "file location of hdf5 grid data")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    std::string filename = vm["data-filename"].as<std::string>();


    // read in relevant data from hdf5 file
    mat X;
    mat Y;
    std::vector<mat> Q_vec(vec_dim);
    {
        H5Easy::File file(filename, H5Easy::File::ReadOnly);
        X = H5Easy::load<mat>(file, "X");
        Y = H5Easy::load<mat>(file, "Y");
        Q_vec[0] = H5Easy::load<mat>(file, "Q1");
        Q_vec[1] = H5Easy::load<mat>(file, "Q2");
        Q_vec[2] = H5Easy::load<mat>(file, "Q3");
        Q_vec[3] = H5Easy::load<mat>(file, "Q4");
        Q_vec[4] = H5Easy::load<mat>(file, "Q5");
    }

    // rescale the X and Y coordinates
    double scale = 1 / std::sqrt(2);
    for (unsigned int i = 0; i < X.size(); ++i)
        for (unsigned int j = 0; j < X[0].size(); ++j)
        {
            X[i][j] *= scale;
            Y[i][j] *= scale;
        }

    // make linear interpolation object
    LinearInterpolation<dim> lp(Q_vec, X, Y);

    // create finite element system to project it onto
    dealii::Triangulation<dim> tria;
    dealii::DoFHandler<dim> dof_handler(tria);
    dealii::FESystem<dim> fe(dealii::FE_Q<dim>(1), vec_dim);
    dealii::Vector<double> solution;
    dealii::AffineConstraints<double> hanging_node_constraints;

    double left = X[0][0];
    double right = X.back()[0];
    double fudge_factor = .9999;
    std::cout << "left is: " << left << "\n";
    std::cout << "right is: " << right << "\n";
    dealii::GridGenerator::hyper_cube(tria, left*fudge_factor, right*fudge_factor);
    tria.refine_global(8);

    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());

    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                    hanging_node_constraints);
    hanging_node_constraints.close();

    dealii::VectorTools::project(dof_handler,
                                 hanging_node_constraints,
                                 dealii::QGauss<dim>(fe.degree + 1),
                                 lp,
                                 solution);

    DirectorPostprocessor<dim> director_postprocessor_defect("cody-data");
    SValuePostprocessor<dim> S_value_postprocessor_defect("cody-data");
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, director_postprocessor_defect);
    data_out.add_data_vector(solution, S_value_postprocessor_defect);
    data_out.build_patches();

    std::cout << "Outputting results" << std::endl;

    std::ofstream output("cody-data.vtu");
    data_out.write_vtu(output);

    return 0;
}
