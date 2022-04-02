#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <vector>
#include <cmath>
#include <iostream>

#include "Numerics/LagrangeMultiplier.hpp"

int main()
{
    const int dim = 3;
    const int vec_dim = 5;
    const int order = 974;

    dealii::Vector<double> Q_vec(vec_dim);
    Q_vec[0] = 0.2;
    Q_vec[3] = 0.3;
    Q_vec[1] = -0.4;
    Q_vec[2] = 0;
    Q_vec[4] = 0;

    dealii::SymmetricTensor<2, dim, double> Q;
    Q[0][0] = Q_vec[0];
    Q[0][1] = Q_vec[1];
    Q[0][2] = Q_vec[2];
    Q[1][1] = Q_vec[3];
    Q[1][2] = Q_vec[4];
    auto eigs = dealii::eigenvectors(Q);

    for (unsigned int i = 0; i < dim; ++i)
        std::cout << eigs[i].first << std::endl;

    std::cout << std::endl;

    dealii::FullMatrix<double> R(dim, dim);
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R[i][j] = eigs[j].second[i];
    R.print(std::cout);
    std::cout << std::endl;

    // Get Lambda
    LagrangeMultiplier<order, dim> lm(1.0, 1e-10, 10);
    dealii::Vector<double> Lambda_vec(vec_dim);
    lm.invertQ(Q_vec);
    lm.returnLambda(Lambda_vec);

    std::cout << Lambda_vec << std::endl;

    dealii::SymmetricTensor<2, dim, double> Lambda;
    Lambda[0][0] = Lambda_vec[0];
    Lambda[0][1] = Lambda_vec[1];
    Lambda[0][2] = Lambda_vec[2];
    Lambda[1][1] = Lambda_vec[3];
    Lambda[1][2] = Lambda_vec[4];
    auto new_eigs = dealii::eigenvectors(Lambda);

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        R[i][j] = eigs[j].second[i];
    R.print(std::cout);

    return 0;
}
