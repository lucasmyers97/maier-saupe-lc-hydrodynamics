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

#include "Numerics/LagrangeMultiplier.hpp"

template <typename Number>
std::pair<std::vector<Number>, unsigned int>
    matrix_to_quaternion(const dealii::Tensor<2, 3, Number> &R)
{
    constexpr int n_quaternion_dofs = 3;
    std::vector<Number> q(n_quaternion_dofs);
    unsigned int first_q_calculated = 0;

    if ((R[0][0] >= 0) && (R[1][1] >=0))
    {
        // calculate real first
        first_q_calculated = 0;
        Number qr = 0.5 * std::sqrt(1 + R[0][0] + R[1][1] + R[2][2]);
        q[0] = (R[2][1] - R[1][2]) / (4 * qr); // qi
        q[1] = (R[0][2] - R[2][0]) / (4 * qr); // qj
        q[2] = (R[1][0] - R[0][1]) / (4 * qr); // qk
    }
    else if ((R[0][0] >= 0) && (R[1][1] < 0))
    {
        // calculate i first
        first_q_calculated = 1;
        Number qi = 0.5 * std::sqrt(1 + R[0][0] - R[1][1] - R[2][2]);
        q[0] = (R[2][1] - R[1][2]) / (4 * qi); // qr
        q[1] = (R[1][0] + R[0][1]) / (4 * qi); // qj
        q[2] = (R[0][2] + R[2][0]) / (4 * qi); // qk
    }
    else if ((R[0][0] < 0) && (R[1][1] >= 0))
    {
        // calculate j first
        first_q_calculated = 2;
        Number qj = 0.5 * std::sqrt(1 - R[0][0] + R[1][1] - R[2][2]);
        q[0] = (R[0][2] - R[2][0]) / (4 * qj); // qr
        q[1] = (R[1][0] + R[0][1]) / (4 * qj); // qi
        q[2] = (R[2][1] + R[1][2]) / (4 * qj); // qk
    }
    else
    {
        // calculate k first
        first_q_calculated = 3;
        Number qk = 0.5 * std::sqrt(1 - R[0][0] - R[1][1] + R[2][2]);
        q[0] = (R[1][0] - R[0][1]) / (4 * qk); // qr
        q[1] = (R[0][2] + R[2][0]) / (4 * qk); // qi
        q[2] = (R[2][1] + R[1][2]) / (4 * qk); // qj
    }

    return std::make_pair(q, first_q_calculated);
}



template <typename Number>
dealii::Tensor<2, 3, Number>
    quaternion_to_matrix(std::vector<Number> q, unsigned int first_q_calculated)
{
    Number qr;
    Number qi;
    Number qj;
    Number qk;

    switch (first_q_calculated)
    {
    case 0:
        qi = q[0];
        qj = q[1];
        qk = q[2];
        qr = std::sqrt(1 - (qi*qi + qj*qj + qk*qk));
        break;
    case 1:
        qr = q[0];
        qj = q[1];
        qk = q[2];
        qi = std::sqrt(1 - (qr * qr + qj * qj + qk * qk));
        break;
    case 2:
        qr = q[0];
        qi = q[1];
        qk = q[2];
        qj = std::sqrt(1 - (qr*qr + qi*qi + qk*qk));
        break;
    case 3:
        qr = q[0];
        qi = q[1];
        qj = q[2];
        qk = std::sqrt(1 - (qr*qr + qi*qi + qj*qj));
        break;
    }

    dealii::Tensor<2, 3, Number> R;

    R[0][0] = 1 - 2 * (qj*qj + qk*qk);
    R[0][1] = 2 * (qi*qj - qk*qr);
    R[0][2] = 2 * (qi*qk + qj*qr);
    R[1][0] = 2 * (qi*qj + qk*qr);
    R[1][1] = 1 - 2 * (qi*qi + qk*qk);
    R[1][2] = 2 * (qj*qk - qi*qr);
    R[2][0] = 2 * (qi*qk - qj*qr);
    R[2][1] = 2 * (qj*qk + qi*qr);
    R[2][2] = 1 - 2 * (qi*qi + qj*qj);

    return R;
}

int main()
{
    const int dim = 3;
    const int vec_dim = 5;
    const int order = 974;

    std::vector<double> Q_vec({.10, 0.06, 0.01, 0.08, 0.01});
    LagrangeMultiplier<dim> lm(order, 1.0, 1e-10, 10);

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
    auto eigs = dealii::eigenvectors(Q);

    dealii::Tensor<2, dim, ADNumberType> R;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R[i][j] = eigs[j].second[i];

    // Need this to make sure it's a rotation matrix
    if (dealii::determinant(R) < 0)
        R *= -1;

    std::cout << std::endl
              << "Printing initial rotation matrix" << R << std::endl
              << std::endl;

    auto quaternion_pair = matrix_to_quaternion(R);
    std::vector<ADNumberType> q = quaternion_pair.first;

    std::cout << "Printing eigenvalues: ";
    for (auto eig : eigs)
        std::cout << eig.first.val() << " ";
    std::cout << std::endl;

    // Need to record first two eigenvalues (since third is determined)
    // Also need three more dofs -- first two come from two components
    std::vector<ADNumberType> eig_dofs(vec_dim);
    eig_dofs[0] = eigs[0].first;
    eig_dofs[1] = eigs[1].first;
    eig_dofs[2] = q[0];
    eig_dofs[3] = q[1];
    eig_dofs[4] = q[2];

    ad_helper.register_dependent_variables(eig_dofs);
    dealii::FullMatrix<double> Jac_input(vec_dim, vec_dim);
    ad_helper.compute_jacobian(Jac_input);

    // Compute LagrangeMultiplier of diagonalized values
    dealii::Vector<double> Q_diag(vec_dim);
    dealii::Vector<double> Lambda_diag(vec_dim);
    dealii::LAPACKFullMatrix<double> Jac(vec_dim, vec_dim);
    Q_diag[0] = eig_dofs[0].val();
    Q_diag[3] = eig_dofs[1].val();
    lm.invertQ(Q_diag);
    lm.returnLambda(Lambda_diag);
    lm.returnJac(Jac);

    // Start inverse transformation
    std::vector<double> new_ind_vars(vec_dim);
    new_ind_vars[0] = Lambda_diag[0];
    new_ind_vars[1] = Lambda_diag[3];
    new_ind_vars[2] = q[0].val();
    new_ind_vars[3] = q[1].val();
    new_ind_vars[4] = q[2].val();

    ad_helper.reset();
    ad_helper.register_independent_variables(new_ind_vars);
    const std::vector<ADNumberType> Lambda_ad
        = ad_helper.get_sensitive_variables();

    // qi = Lambda_ad[2];
    // qj = Lambda_ad[3];
    // qr = Lambda_ad[4];
    // qk = std::sqrt(1 - (qi*qi + qj*qj + qr*qr));

    // R[0][0] = 1 - 2*qj*qj - 2*qk*qk;
    // R[0][1] = 2 * (qi*qj - qk*qr);
    // R[0][2] = 2 * (qi*qk + qj*qr);
    // R[1][0] = 2 * (qi*qj + qk*qr);
    // R[1][1] = 1 - 2*qi*qi - 2*qk*qk;
    // R[1][2] = 2 * (qj*qk - qi*qr);
    // R[2][0] = 2 * (qi*qk - qj*qr);
    // R[2][1] = 2 * (qj*qk + qi*qr);
    // R[2][2] = 1 - 2*qi*qi - 2*qj*qj;

    q[0] = Lambda_ad[2];
    q[1] = Lambda_ad[3];
    q[2] = Lambda_ad[4];

    R = quaternion_to_matrix(q, quaternion_pair.second);

    std::cout << std::endl << "Printing new rotation matrix "
              << R << std::endl << std::endl;

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

    std::cout << std::endl << "Printing non-diag Lambda "
              << Lambda << std::endl;

    Jac.invert();
    Jac(1, 0) = Jac(3, 0);
    Jac(0, 1) = Jac(0, 3);
    Jac(1, 1) = Jac(3, 3);
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
    tmp1.print(std::cout, 10, 3);
    std::cout << std::endl;
    Jac_output.mmult(tmp2, tmp1);
    tmp2.mmult(Jac_new, Jac_input);
    std::cout << "printing composition of Jacobians" << std::endl;
    Jac_new.print(std::cout, 20, 6);
    std::cout << std::endl;

    Jac_output.mmult(tmp1, Jac_input);
    std::cout << "Printing input mult output Jacobians" << std::endl;
    for (unsigned int i = 0; i < vec_dim; ++i)
        for (unsigned int j = 0; j < vec_dim; ++j)
            if (tmp1[i][j] < 1e-15)
                tmp1[i][j] = 0;
    tmp1.print(std::cout, 20, 6);
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
    std::cout << std::endl << "Printing difference between differently-calculated Lambdas: "
              << Lambda << std::endl << std::endl;
    std::cout << std::endl << "Printing non_diag Q: "
              << Q_non_diag << std::endl << std::endl;
    dealii::FullMatrix<double> tmp(vec_dim, vec_dim);
    Jac_non_diag.invert();
    tmp = Jac_non_diag;
    for (unsigned int i = 0; i < vec_dim; ++i)
        for (unsigned int j = 0; j < vec_dim; ++j)
            if (std::abs(tmp[i][j]) < 1e-14)
                tmp[i][j] = 0;
    tmp.print(std::cout, 20, 6);

    dealii::FullMatrix<double> Jac_diff(vec_dim, vec_dim);
    for (unsigned int i = 0; i < vec_dim; ++i)
        for (unsigned int j = 0; j < vec_dim; ++j)
            std::cout << tmp(i, j) - Jac_new(i, j) << std::endl;

    return 0;
}
