#include <deal.II/differentiation/ad.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <vector>
#include <cmath>
#include <iostream>

#include "Numerics/LagrangeMultiplier.hpp"

int main()
{
    const int dim = 3;
    const int vec_dim = 5;
    const int order = 974;

    std::vector<double> Q_vec({0.15, 0.25, 0, 0.15, 0});
    // std::vector<double> Q_vec({0.45, 0, 0, 0.15, 0});
    LagrangeMultiplier<order, dim> lm(1.0, 1e-10, 10);

    // set names of things
    constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
        dealii::Differentiation::AD::NumberTypes::sacado_dfad;
    using ADHelper =
        dealii::Differentiation::AD::VectorFunction<dim, ADTypeCode, double>;
    using ADNumberType = typename ADHelper::ad_type;

    // set up automatic differentiation
    ADHelper ad_helper(vec_dim, vec_dim);
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
    std::array<std::pair<ADNumberType,
                         dealii::Tensor<1, dim, ADNumberType>>, dim>
        eigs = dealii::eigenvectors(Q);

    dealii::Tensor<2, dim, ADNumberType> R;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R[i][j] = eigs[j].second[i];

    std::cout << R << std::endl << std::endl;

    ADNumberType qk = 0.5 * std::sqrt(1 - R[0][0] - R[1][1] + R[2][2]);
    ADNumberType qi = (R[0][2] + R[2][0]) / (4 * qk);
    ADNumberType qj = (R[2][1] + R[1][2]) / (4 * qk);
    ADNumberType qr = (R[1][0] - R[0][1]) / (4 * qk);

    // Need to record first two eigenvalues (since third is determined)
    // Also need three more dofs -- first two come from two components
    std::vector<ADNumberType> eig_dofs(vec_dim);
    eig_dofs[0] = eigs[0].first;
    eig_dofs[1] = eigs[1].first;
    eig_dofs[2] = qi;
    eig_dofs[3] = qj;
    eig_dofs[4] = qr;

    ad_helper.register_dependent_variables(eig_dofs);
    dealii::FullMatrix<double> Jac_input(vec_dim, vec_dim);
    ad_helper.compute_jacobian(Jac_input);
    Jac_input.print(std::cout);
    std::cout << std::endl;

    // Compute LagrangeMultiplier of diagonalized values
    dealii::Vector<double> Q_diag(vec_dim);
    dealii::Vector<double> Lambda_diag(vec_dim);
    dealii::LAPACKFullMatrix<double> Jac(vec_dim, vec_dim);
    Q_diag[0] = eig_dofs[0].val();
    Q_diag[3] = eig_dofs[1].val();
    lm.invertQ(Q_diag);
    lm.returnLambda(Lambda_diag);
    lm.returnJac(Jac);

    std::vector<double> new_ind_vars(vec_dim);
    new_ind_vars[0] = Lambda_diag[0];
    new_ind_vars[1] = Lambda_diag[3];
    new_ind_vars[2] = qi.val();
    new_ind_vars[3] = qj.val();
    new_ind_vars[4] = qr.val();

    ad_helper.reset();
    ad_helper.register_independent_variables(new_ind_vars);
    const std::vector<ADNumberType> Lambda_ad = ad_helper.get_sensitive_variables();

    qi = Lambda_ad[2];
    qj = Lambda_ad[3];
    qr = Lambda_ad[4];
    qk = std::sqrt(1 - (qi*qi + qj*qj + qr*qr));

    R[0][0] = 1 - 2*qj*qj - 2*qk*qk;
    R[0][1] = 2 * (qi*qj - qk*qr);
    R[0][2] = 2 * (qi*qk + qj*qr);
    R[1][0] = 2 * (qi*qj + qk*qr);
    R[1][1] = 1 - 2*qi*qi - 2*qk*qk;
    R[1][2] = 2 * (qj*qk - qi*qr);
    R[2][0] = 2 * (qi*qk - qj*qr);
    R[2][1] = 2 * (qj*qk + qi*qr);
    R[2][2] = 1 - 2*qi*qi - 2*qj*qj;

    std::cout << R << std::endl << std::endl;

    // Undiagonalize Lambda_diag
    dealii::SymmetricTensor<2, dim, ADNumberType> Lambda_mat;
    Lambda_mat[0][0] = Lambda_ad[0];
    Lambda_mat[1][1] = Lambda_ad[1];
    Lambda_mat[2][2] = -(Lambda_ad[0] + Lambda_ad[1]);

    dealii::Tensor<2, dim, ADNumberType> Lambda_tens
        = R * Lambda_mat * dealii::transpose(R);

    std::vector<ADNumberType> new_dofs(vec_dim);
    new_dofs[0] = Lambda_tens[0][0];
    new_dofs[1] = Lambda_tens[0][1];
    new_dofs[2] = Lambda_tens[0][2];
    new_dofs[3] = Lambda_tens[1][1];
    new_dofs[4] = Lambda_tens[1][2];

    dealii::Vector<double> Lambda(vec_dim);
    dealii::FullMatrix<double> Jac_output(vec_dim, vec_dim);
    ad_helper.register_dependent_variables(new_dofs);
    ad_helper.compute_values(Lambda);
    ad_helper.compute_jacobian(Jac_output);

    std::cout << Lambda << std::endl;
    Jac_output.print(std::cout);
    std::cout << std::endl;

    for (unsigned int i = 0; i < vec_dim; ++i)
        for (unsigned int j = 0; j < vec_dim; ++j)
        {
            if ((i < 2) && (j < 2))
                continue;
            else if (i == j)
                Jac(i, j) = 1;
            else
                Jac(i, j) = 0;
        }

    dealii::FullMatrix<double> Jac_new(vec_dim, vec_dim);
    dealii::FullMatrix<double> tmp1(vec_dim, vec_dim);
    dealii::FullMatrix<double> tmp2(vec_dim, vec_dim);
    tmp1 = Jac;
    std::cout << "printing Jac" << std::endl;
    tmp1.print(std::cout);
    std::cout << std::endl;
    Jac_output.mmult(tmp2, tmp1);
    tmp2.mmult(Jac_new, Jac_input);
    Jac_new.print(std::cout, 10, 3);
    std::cout << std::endl;

    dealii::Vector<double> Q_non_diag(vec_dim);
    for (unsigned int i = 0; i < vec_dim; ++i)
        Q_non_diag[i] = Q_vec[i];
    dealii::Vector<double> Lambda_non_diag(vec_dim);
    dealii::LAPACKFullMatrix<double> Jac_non_diag(vec_dim, vec_dim);
    lm.invertQ(Q_non_diag);
    lm.returnLambda(Lambda_non_diag);
    lm.returnJac(Jac_non_diag);

    Lambda -= Lambda_non_diag;
    std::cout << Lambda << std::endl << std::endl;
    std::cout << Q_non_diag << std::endl << std::endl;
    dealii::FullMatrix<double> tmp(vec_dim, vec_dim);
    tmp = Jac_non_diag;
    tmp.print(std::cout, 10, 3);

    return 0;
}
