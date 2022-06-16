#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <highfive/H5Easy.hpp>

#include <boost/program_options.hpp>

#include <deal.II/lac/vector.h>

#include "Numerics/LagrangeMultiplier.hpp"

#include <iostream>
#include <string>

int main(int ac, char* av[])
{
    using mat = std::vector<std::vector<double>>;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("folder", po::value<std::string>(), "folder of data file")
        ("filename", po::value<std::string>(), "name of data file")
        ("tol", po::value<double>(), "tolerance of newton's method")
        ("max_iters", po::value<int>(), "max number of newton iterations")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
    }

    std::string folder = vm["folder"].as<std::string>();
    std::string input_filename = folder + vm["filename"].as<std::string>();
    std::string output_filename = folder + "lambda_vals.h5";

    double tol = vm["tol"].as<double>();
    int max_iter = vm["max_iters"].as<int>();

    mat Q1;
    mat Q2;
    mat Q3;
    mat Q4;
    mat Q5;
    mat X;
    mat Y;
    {
    H5Easy::File file(input_filename, H5Easy::File::ReadOnly);
    Q1 = H5Easy::load<mat>(file, "Q1");
    Q2 = H5Easy::load<mat>(file, "Q2");
    Q3 = H5Easy::load<mat>(file, "Q3");
    Q4 = H5Easy::load<mat>(file, "Q4");
    Q5 = H5Easy::load<mat>(file, "Q5");
    X = H5Easy::load<mat>(file, "X");
    Y = H5Easy::load<mat>(file, "Y");
    }

    int m = Q1.size();
    int n = Q1[0].size();
    mat Lambda1(m, std::vector<double>(n));
    mat Lambda2(m, std::vector<double>(n));
    mat Lambda3(m, std::vector<double>(n));
    mat Lambda4(m, std::vector<double>(n));
    mat Lambda5(m, std::vector<double>(n));
    dealii::Vector<double> Q_vec(5);
    dealii::Vector<double> Lambda(5);
    LagrangeMultiplier<3> lagrange_multiplier(590, 1, tol, max_iter);
    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
        {
            Q_vec[0] = Q1[i][j];
            Q_vec[1] = Q2[i][j];
            Q_vec[2] = Q2[i][j];
            Q_vec[3] = Q4[i][j];
            Q_vec[4] = Q5[i][j];

            lagrange_multiplier.invertQ(Q_vec);
            lagrange_multiplier.returnLambda(Lambda);

            Lambda1[i][j] = Lambda[0];
            Lambda2[i][j] = Lambda[1];
            Lambda2[i][j] = Lambda[2];
            Lambda4[i][j] = Lambda[3];
            Lambda5[i][j] = Lambda[4];
        }

    {
    H5Easy::File file(output_filename, H5Easy::File::Overwrite);
    H5Easy::dump(file, "Lambda1", Lambda1);
    H5Easy::dump(file, "Lambda2", Lambda2);
    H5Easy::dump(file, "Lambda3", Lambda3);
    H5Easy::dump(file, "Lambda4", Lambda4);
    H5Easy::dump(file, "Lambda5", Lambda5);
    H5Easy::dump(file, "X", X);
    H5Easy::dump(file, "Y", Y);
    }

    return 0;
}
