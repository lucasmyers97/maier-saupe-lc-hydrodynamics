#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <vector>

#include "Utilities/maier_saupe_constants.hpp"
#include "Numerics/LagrangeMultiplierReduced.hpp"
#include "Numerics/LagrangeMultiplier.hpp"

namespace msc = maier_saupe_constants;

int main()
{
    const int dim = 3;
    const int order = 974;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const int max_iters = 20;
    const int num = 100000;

    const double epsilon = 1e-2;

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    Q_vec[0] = 0.08 + epsilon;
    Q_vec[1] = 0.01;
    Q_vec[2] = 0.06;
    Q_vec[3] = 0.08;
    Q_vec[4] = 0.02;

    for (unsigned int i = 0; i < num; ++i)
    {
    // Calculate eigenvectors and eigenvalues
    dealii::SymmetricTensor<2, dim, double> Q_mat;
    Q_mat[0][0] = Q_vec[0];
    Q_mat[0][1] = Q_vec[1];
    Q_mat[0][2] = Q_vec[2];
    Q_mat[1][1] = Q_vec[3];
    Q_mat[1][2] = Q_vec[4];
    Q_mat[2][2] = -(Q_mat[0][0] + Q_mat[1][1]);

    auto eigs = dealii::eigenvectors(Q_mat);

    dealii::Tensor<2, dim, double> R;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R[i][j] = eigs[j].second[i];

    // Calculate first transformation jacobian
    dealii::FullMatrix<double> dlambda(2, msc::vec_dim<dim>);
    for (unsigned int i = 0; i < 2; ++i)
    {
        dlambda(i, 0) = R[0][i] * R[0][i] - R[2][i] * R[2][i];
        dlambda(i, 1) = R[0][i] * R[1][i] + R[1][i] * R[0][i];
        dlambda(i, 2) = R[0][i] * R[2][i] + R[2][i] * R[0][i];
        dlambda(i, 3) = R[1][i] * R[1][i] - R[2][i] * R[2][i];
        dlambda(i, 4) = R[1][i] * R[2][i] + R[2][i] * R[1][i];
    }

    std::vector<dealii::FullMatrix<double>>
        S(3, dealii::FullMatrix<double>(dim, msc::vec_dim<dim>));
    std::vector<double> dB_ni_nj(msc::vec_dim<dim>);
    for (unsigned int l = 0; l < 3; ++l)
    {
        int i = (l < 2) ? 0 : 1;
        int j = (l < 1) ? 1 : 2;

        dB_ni_nj[0] = R[0][i]*R[0][j] - R[2][i]*R[2][j];
        dB_ni_nj[1] = R[0][i]*R[1][j] + R[1][i]*R[0][j];
        dB_ni_nj[2] = R[0][i]*R[2][j] + R[2][i]*R[0][j];
        dB_ni_nj[3] = R[1][i]*R[1][j] - R[2][i]*R[2][j];
        dB_ni_nj[4] = R[1][i]*R[2][j] + R[2][i]*R[1][j];

        for (unsigned int k = 0; k < dim; ++k)
            for (unsigned int m = 0; m < msc::vec_dim<dim>; ++m)
                S[l](k, m) = R[k][j] * dB_ni_nj[m];
    }

    std::vector<double> gamma(3);
    if ((eigs[0].first - eigs[1].first) != 0)
        gamma[0] = 1 / (eigs[0].first - eigs[1].first);
    gamma[1] = 1 / (eigs[0].first - eigs[2].first);
    gamma[2] = 1 / (eigs[1].first - eigs[2].first);

    // Calculate middle transformation jacobian
    dealii::Tensor<1, 2, double> Q_red;
    Q_red[0] = eigs[0].first;
    Q_red[1] = eigs[1].first;

    dealii::Tensor<1, 2, double> Lambda_red;
    dealii::Tensor<2, 2, double> Jac_red;
    LagrangeMultiplierReduced lmr(order, alpha, tol, max_iters);
    lmr.invertQ(Q_red);
    Lambda_red = lmr.returnLambda();
    Jac_red = lmr.returnJac();
    Jac_red = dealii::invert(Jac_red);

    dealii::FullMatrix<double> dLambda(2, 2);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        dLambda(i, j) = Jac_red[i][j];

    // Calculate final transformation jacobian
    std::vector<dealii::FullMatrix<double>>
        T(2, dealii::FullMatrix<double>(msc::vec_dim<dim>, dim));
    for (unsigned int i = 0; i < 2; ++i)
    {
        T[i](0, 0) = 2 * R[0][i];
        T[i](1, 0) = R[1][i];
        T[i](2, 0) = R[2][i];
        T[i](1, 1) = R[0][i];
        T[i](3, 1) = 2 * R[1][i];
        T[i](4, 1) = R[2][i];
        T[i](2, 2) = R[0][i];
        T[i](4, 2) = R[1][i];
    }

    dealii::FullMatrix<double> dF(msc::vec_dim<dim>, 2);
    for (unsigned int i = 0; i < 2; ++i) {
      dF(0, i) = R[0][i] * R[0][i] - R[0][2] * R[0][2];
      dF(1, i) = R[0][i] * R[1][i] - R[0][2] * R[1][2];
      dF(2, i) = R[0][i] * R[2][i] - R[0][2] * R[2][2];
      dF(3, i) = R[1][i] * R[1][i] - R[1][2] * R[1][2];
      dF(4, i) = R[1][i] * R[2][i] - R[1][2] * R[2][2];
    }

    // Multiply all of the jacobians together
    std::vector<dealii::FullMatrix<double>> TS(
        3, dealii::FullMatrix<double>(msc::vec_dim<dim>, msc::vec_dim<dim>));
    T[0].mmult(TS[0], S[0]);
    T[0].mmult(TS[1], S[1]);
    T[1].mmult(TS[2], S[2]);

    dealii::FullMatrix<double> Jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    Jac.triple_product(dLambda, dF, dlambda);
    Jac.add((Lambda_red[0] - Lambda_red[1]) * gamma[0], TS[0],
            (2 * Lambda_red[0] + Lambda_red[1]) * gamma[1], TS[1],
            (Lambda_red[0] + 2 * Lambda_red[1]) * gamma[2], TS[2]);
    // Jac.add((dLambda(0, 0) - dLambda(0, 1)), TS[0],
    //         (3 * Lambda_red[0] + epsilon * (2*dLambda(0, 0) + dLambda(0, 1))) * gamma[1], TS[1],
    //         (3 * Lambda_red[0] + epsilon * (dLambda(0, 0) + 2*dLambda(0, 1))) * gamma[2], TS[2]);
    // Jac.add((dLambda(0, 0) - dLambda(0, 1)), TS[0],
    //         3 * Lambda_red[0] * gamma[1], TS[1],
    //         3 * Lambda_red[0] * gamma[2], TS[2]);

    // Jac.print(std::cout, 15, 5);
    // std::cout << "\n";

    // Compare against original-frame calculation
    // LagrangeMultiplier<order> lm(alpha, tol, max_iters);
    // lm.invertQ(Q_vec);
    // dealii::LAPACKFullMatrix<double> lapack_jac;
    // dealii::FullMatrix<double> regular_jac(msc::vec_dim<dim>,
    //                                        msc::vec_dim<dim>);
    // lm.returnJac(lapack_jac);
    // lapack_jac.invert();
    // regular_jac = lapack_jac;

    // for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
    //     for (unsigned int j = 0; j < msc::vec_dim<dim>; ++j)
    //         Jac(i, j) -= regular_jac(i, j);

    // Jac.print(std::cout, 15, 5);
    // std::cout << "\n";
    // regular_jac.print(std::cout, 15, 5);
    }
    return 0;
}
