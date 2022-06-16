#include "Numerics/LagrangeMultiplier.hpp"

#include <deal.II/base/point.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <iostream>

int main()
{
    const int dim = 3;
    const int vec_dim = 5;

    double alpha{1};
    double tol{1e-12};
    unsigned int max_iter{50};
    constexpr int order = 2702;

    dealii::Vector<double> Q_vec(vec_dim);
    Q_vec[0] = 0.45006666666668388;
    Q_vec[1] = 0;
    Q_vec[2] = 0;
    Q_vec[3] = -0.22503333333334197;
    Q_vec[4] = -0.33333333333334603;

    LagrangeMultiplier<dim> l(order, alpha, tol, max_iter);
    l.invertQ(Q_vec);

    dealii::Vector<double> Lambda(vec_dim);
    dealii::FullMatrix<double> Jac(vec_dim, vec_dim);
    dealii::LAPACKFullMatrix<double> lapack_Jac(vec_dim, vec_dim);
    double Z;

    l.returnLambda(Lambda);
    l.returnJac(lapack_Jac);
    Z = l.returnZ();

    Jac = lapack_Jac;

    std::cout << Lambda << "\n\n";
    Jac.print(std::cout, 15, 5);
    std::cout << "\n" << Z << "\n\n";

    return 0;
}
