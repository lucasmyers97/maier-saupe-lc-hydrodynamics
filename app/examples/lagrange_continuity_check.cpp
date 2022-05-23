#include "Numerics/LagrangeMultiplier.hpp"

#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <highfive/H5Easy.hpp>

#include <iostream>
#include <vector>

int main()
{
    const int order = 974;
    const int num = 1000;
    const double eps_max = 0.001;

    double alpha = 1.0;
    double tol = 1e-12;
    int max_iters = 50;

    std::vector<std::vector<double>>
        Lagrange_array(num, std::vector<double>(5));
    std::vector<std::vector<std::vector<double>>>
        Jac_array(num, std::vector<std::vector<double>>(5, std::vector<double>(5)));

    double eps = 0;
    for (unsigned int i = 0; i < num; ++i)
    {
        eps = (i - static_cast<double>(num) / 2)
              * (eps_max / static_cast<double>(num));

        dealii::Vector<double> Q({6.0 / 10.0 + eps, 0.0, 0.0, -3.0 / 10.0, 0.0});
        dealii::Vector<double> Lambda;
        Lambda.reinit(5);
        dealii::LAPACKFullMatrix<double> Jac(5, 5);

        LagrangeMultiplier<order> lm(alpha, tol, max_iters);
        lm.invertQ(Q);
        lm.returnLambda(Lambda);
        lm.returnJac(Jac);

        for (unsigned int j = 0; j < 5; ++j)
        {
            Lagrange_array[i][j] = Lambda[j];

            for (unsigned int k = 0; k < 5; ++k)
                Jac_array[i][j][k] = Jac(j, k);
        }
    }

    H5Easy::File file("lagrange_data.h5", H5Easy::File::Overwrite);
    H5Easy::dump(file, "/Lagrange_array", Lagrange_array);
    H5Easy::dump(file, "/Jac_array", Jac_array);

    return 0;
}
