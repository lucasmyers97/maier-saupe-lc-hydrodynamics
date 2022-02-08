#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <vector>
#include <cmath>
#include <string>
#include <highfive/H5Easy.hpp>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/option.hpp>
#include "Numerics/LagrangeMultiplier.hpp"

using mat = dealii::LAPACKFullMatrix<double>;
using vec = dealii::Vector<double>;
namespace po = boost::program_options;

int main(int ac, char** av)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("data_path", po::value<std::string>(), "set data path")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    std::string filedir;
    if (vm.count("data_path"))
    {
        std::cout << "data_path is: " << vm["data_path"].as<std::string>() << "\n";
        filedir = vm["data_path"].as<std::string>();
    } else {
        filedir =".";
    }
    std::string filename = filedir + "/data.h5";
    H5Easy::File file(filename, H5Easy::File::Overwrite);

    const int vec_dim = 5;
    const int mat_dim = 3;
    const int n_pts = 100;
    const double S_max = 0.7;
    const double S_min = -0.1;
    const int lebedev_order = 590;
    const double alpha = 1.0;
    const double tol = 1e-12;
    const int max_iter = 12;

    mat Q(mat_dim, mat_dim);
    vec Q_vec(vec_dim);
    mat Q_squared(mat_dim, mat_dim);
    double trace_Q_squared = 0;
    mat Lambda(mat_dim, mat_dim);
    mat Lambda_Q(mat_dim, mat_dim);
    vec Lambda_vec(vec_dim);
    double trace_Lambda_Q = 0;
    LagrangeMultiplier<lebedev_order> lm(alpha, tol, max_iter);

    double log_4_pi = std::log(4*M_PI);
    double log_Z = 0;
    constexpr int n_kappas = 3;
    double Kappa[n_kappas] = {3.3, 3.4049, 3.5};

    std::vector<double> f(n_pts);
    std::vector<double> S_vec(n_pts);

    double S = 0;
    double Z = 0;
    for (int j = 0; j < n_kappas; ++j)
    {
        for (int i = 0; i < n_pts; ++i)
        {
            S = i * ((S_max - S_min) / (n_pts - 1)) + S_min;
            Q(0, 0) = (2.0/3.0) * S;
            Q(0, 1) = 0;
            Q(0, 2) = 0;
            Q(1, 0) = 0;
            Q(1, 1) = -(1.0/3.0) * S;
            Q(1, 2) = 0;
            Q(2, 0) = 0;
            Q(2, 1) = 0;
            Q(2, 2) = -(1.0/3.0) * S;

            Q_vec(0) = Q(0, 0);
            Q_vec(3) = Q(1, 1);
            lm.invertQ(Q_vec);
            lm.returnLambda(Lambda_vec);
            Z = lm.returnZ();
            // Z = 1.0;

            Lambda(0, 0) = Lambda_vec(0);
            Lambda(0, 1) = 0;
            Lambda(0, 2) = 0;
            Lambda(1, 0) = 0;
            Lambda(1, 1) = Lambda_vec(1);
            Lambda(1, 2) = 0;
            Lambda(2, 0) = 0;
            Lambda(2, 1) = 0;
            Lambda(2, 2) = -(Lambda(0, 0) + Lambda(1, 1));

            Q.mmult(Q_squared, Q);
            // Q_squared.print_formatted(std::cout);
            trace_Q_squared = Q_squared.trace();

            Lambda.mmult(Lambda_Q, Q);
            trace_Lambda_Q = Lambda_Q.trace();

            log_Z = std::log(Z);

            f[i] = -Kappa[j]*trace_Q_squared + log_4_pi - log_Z + trace_Lambda_Q;
            S_vec[i] = S;
        }

        H5Easy::dump(file, "/free_energy_" + std::to_string(Kappa[j]), f);
    }
    
    H5Easy::dump(file, "/S_val", S_vec);

    return 0;
}
