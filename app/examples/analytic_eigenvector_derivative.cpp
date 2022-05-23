#include <deal.II/differentiation/ad.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <vector>
#include <cmath>
#include <iostream>
#include <utility>

int main()
{
    const int dim = 3;
    const int vec_dim = 5;
    const int output_dim = 11;

    std::vector<double> Q_vec({.10, 0.06, 0.08, 0, 0});

    // set names of things
    constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
        dealii::Differentiation::AD::NumberTypes::sacado_dfad;
    using ADHelper =
        dealii::Differentiation::AD::VectorFunction<dim, ADTypeCode, double>;
    using ADNumberType = typename ADHelper::ad_type;

    // set up automatic differentiation
    ADHelper ad_helper(vec_dim, output_dim);
    ad_helper.register_independent_variables(Q_vec);
    const std::vector<ADNumberType> Q_ad
        = ad_helper.get_sensitive_variables();

    // diagonalize and keep track of eigen-numbers
    dealii::SymmetricTensor<2, dim, ADNumberType> Q;
    Q[0][0] = Q_ad[0];
    Q[0][1] = Q_ad[1];
    Q[0][2] = Q_ad[2];
    Q[1][1] = Q_ad[3];
    Q[1][2] = Q_ad[4];
    Q[2][2] = -(Q_ad[0] + Q_ad[3]);
    auto eigs = dealii::eigenvectors(Q);

    dealii::Tensor<2, dim, ADNumberType> R;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R[i][j] = eigs[j].second[i];

    // Need this to make sure it's a rotation matrix
    if (dealii::determinant(R) < 0)
        R *= -1;

    std::cout << std::endl
              << "Printing initial rotation matrix\n"
              << R << std::endl
              << std::endl;

    std::vector<ADNumberType> outputs(output_dim);
    outputs[0] = eigs[0].first;
    outputs[1] = eigs[1].first;
    outputs[2] = R[0][0];
    outputs[3] = R[1][0];
    outputs[4] = R[2][0];
    outputs[5] = R[0][1];
    outputs[6] = R[1][1];
    outputs[7] = R[2][1];
    outputs[8] = R[0][2];
    outputs[9] = R[1][2];
    outputs[10] = R[2][2];

    dealii::Vector<double> outputs_vec;
    dealii::FullMatrix<double> outputs_jac;

    ad_helper.register_dependent_variables(outputs);
    ad_helper.compute_values(outputs_vec);
    ad_helper.compute_jacobian(outputs_jac);

    std::cout << outputs_vec << "\n\n";
    outputs_jac.print(std::cout);
    std::cout << "\n";

    // set up derivative of Q
    std::vector<dealii::SymmetricTensor<2, dim>> dB(vec_dim);
    dB[0][0][0] = 1;
    dB[0][2][2] = -1;
    dB[1][0][1] = 1;
    dB[2][0][2] = 1;
    dB[3][1][1] = 1;
    dB[3][2][2] = -1;
    dB[4][1][2] = 1;

    std::vector<double> lambda(dim);
    lambda[0] = outputs_vec[0];
    lambda[1] = outputs_vec[1];
    lambda[2] = -(lambda[0] + lambda[1]);

    std::vector<dealii::Tensor<1, dim>> n(dim);
    n[0][0] = outputs_vec[2];
    n[0][1] = outputs_vec[3];
    n[0][2] = outputs_vec[4];
    n[1][0] = outputs_vec[5];
    n[1][1] = outputs_vec[6];
    n[1][2] = outputs_vec[7];
    n[2][0] = outputs_vec[8];
    n[2][1] = outputs_vec[9];
    n[2][2] = outputs_vec[10];

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dn(vec_dim, std::vector<dealii::Tensor<1, dim>>(dim));

    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int k = 0; k < vec_dim; ++k)
                if (i != j)
                    dn[k][i] += 1 / (lambda[i] - lambda[j]) * ((dB[k] * n[i]) * n[j]) * n[j];

    dealii::FullMatrix<double> Jac_analytic(11, vec_dim);
    for (unsigned int k = 0; k < vec_dim; ++k)
    {
        Jac_analytic(0, k) = (dB[k] * n[0]) * n[0];
        Jac_analytic(1, k) = (dB[k] * n[1]) * n[1];
        Jac_analytic(2, k) = dn[k][0][0];
        Jac_analytic(3, k) = dn[k][0][1];
        Jac_analytic(4, k) = dn[k][0][2];
        Jac_analytic(5, k) = dn[k][1][0];
        Jac_analytic(6, k) = dn[k][1][1];
        Jac_analytic(7, k) = dn[k][1][2];
        Jac_analytic(8, k) = dn[k][2][0];
        Jac_analytic(9, k) = dn[k][2][1];
        Jac_analytic(10, k) = dn[k][2][2];
    }

    Jac_analytic.print(std::cout);

    return 0;
}
