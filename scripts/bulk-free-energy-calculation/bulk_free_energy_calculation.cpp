#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <vector>
#include <cmath>
#include <string>
#include <highfive/H5Easy.hpp>
#include "LagrangeMultiplier.hpp"

using mat = dealii::LAPACKFullMatrix<double>;
using vec = dealii::Vector<double>;

int main()
{
    const int vec_dim = 5;
    const int mat_dim = 3;
    const int n_pts = 100;
    const double S_max = 0.65;
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
    double Kappa = 3.4049;

    std::vector<double> f(n_pts);
    std::vector<double> S_vec(n_pts);

    double S = 0;
    double Z = 0;
    for (int i = 0; i < n_pts; ++i)
    {
        S = i * (S_max / (n_pts - 1));
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
        lm.setQ(Q_vec);
        lm.returnLambda(Lambda_vec);
        Z = lm.calcZ();

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

        f[i] = -Kappa*trace_Q_squared + log_4_pi - log_Z + trace_Lambda_Q;
        S_vec[i] = S;

        // if (i == 0)
        // {
        //     std::cout << trace_Q_squared << "\n";
        //     std::cout << log_Z << "\n";
        //     std::cout << log_4_pi << "\n";
        //     std::cout << trace_Lambda_Q << "\n";
        // }
    }

    std::string filename = "/home/lucasmyers97/maier-saupe-lc-hydrodynamics/data/bulk-free-energy-calculation/2021-10-08/data.h5";
    H5Easy::File file(filename, H5Easy::File::Overwrite);
    H5Easy::dump(file, "/free_energy", f);
    H5Easy::dump(file, "/S_val", S_vec);

    return 0;
}
