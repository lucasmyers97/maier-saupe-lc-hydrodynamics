#include <deal.II/differentiation/ad.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <vector>
#include <cmath>
#include <iostream>


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

    std::vector<double> Q_vec({0.15, 0.25, 0.05, 0.15, 0});

    // set names of things
    constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
        dealii::Differentiation::AD::NumberTypes::sacado_dfad;
    using ADHelper =
        dealii::Differentiation::AD::VectorFunction<dim, ADTypeCode, double>;
    using ADNumberType = typename ADHelper::ad_type;

    // set up automatic differentiation
    ADHelper ad_helper(vec_dim, vec_dim);
    ad_helper.register_independent_variables(Q_vec);
    std::vector<ADNumberType> Q_ad
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

    // Make sure it's a rotation matrix
    if (dealii::determinant(R).val() < 0)
        R *= -1;

    // Print everything out
    std::cout << "Printing determinate: " << dealii::determinant(R) << std::endl;
    std::cout << std::endl << "Printing diagonalized Q" << std::endl;
    std::cout << dealii::transpose(R) * Q * R << std::endl << std::endl;
    std::cout << std::endl << "Printing rotation" << std::endl;
    std::cout << R << std::endl << std::endl;

    auto quaternion_pair = matrix_to_quaternion(R);
    std::vector<ADNumberType> q = quaternion_pair.first;

    // Need to record first two eigenvalues (since third is determined)
    // Also need three more dofs -- first two come from two components
    std::vector<ADNumberType> eig_dofs(vec_dim);
    eig_dofs[0] = eigs[0].first;
    eig_dofs[1] = eigs[1].first;
    eig_dofs[2] = q[0];
    eig_dofs[3] = q[1];
    eig_dofs[4] = q[2];

    // Get Jacobian from transformation from R^5 -> R^2 x R^3
    ad_helper.register_dependent_variables(eig_dofs);
    dealii::FullMatrix<double> Jac_input(vec_dim, vec_dim);
    ad_helper.compute_jacobian(Jac_input);

    // Now start the inverse transformation
    dealii::Vector<double> Q_diag(vec_dim);
    Q_diag[0] = eig_dofs[0].val();
    Q_diag[3] = eig_dofs[1].val();

    std::vector<double> new_ind_vars(vec_dim);
    new_ind_vars[0] = Q_diag[0];
    new_ind_vars[1] = Q_diag[3];
    new_ind_vars[2] = q[0].val();
    new_ind_vars[3] = q[1].val();
    new_ind_vars[4] = q[2].val();

    ad_helper.reset();
    ad_helper.register_independent_variables(new_ind_vars);
    Q_ad = ad_helper.get_sensitive_variables();

    q[0] = Q_ad[2];
    q[1] = Q_ad[3];
    q[2] = Q_ad[4];

    R = quaternion_to_matrix(q, quaternion_pair.second);

    std::cout << std::endl << "Printing quaternions "
              << q[0] << " " << q[1] << " " << q[2] <<  std::endl;

    // R[0][0] = 1 - 2*qj*qj - 2*qk*qk;
    // R[0][1] = 2 * (qi*qj - qk*qr);
    // R[0][2] = 2 * (qi*qk + qj*qr);
    // R[1][0] = 2 * (qi*qj + qk*qr);
    // R[1][1] = 1 - 2*qi*qi - 2*qk*qk;
    // R[1][2] = 2 * (qj*qk - qi*qr);
    // R[2][0] = 2 * (qi*qk - qj*qr);
    // R[2][1] = 2 * (qj*qk + qi*qr);
    // R[2][2] = 1 - 2*qi*qi - 2*qj*qj;

    std::cout << std::endl << "Printing new rotation matrix "
              << R << std::endl << std::endl;

    // Undiagonalize Q_diag
    Q.clear();
    Q[0][0] = Q_ad[0];
    Q[1][1] = Q_ad[1];
    Q[2][2] = -(Q_ad[0] + Q_ad[1]);

    dealii::Tensor<2, dim, ADNumberType> Q_tens = R * Q * dealii::transpose(R);

    // Register output with ad_helper
    std::vector<ADNumberType> new_dofs(vec_dim);
    new_dofs[0] = Q_tens[0][0];
    new_dofs[1] = Q_tens[0][1];
    new_dofs[2] = Q_tens[0][2];
    new_dofs[3] = Q_tens[1][1];
    new_dofs[4] = Q_tens[1][2];

    dealii::FullMatrix<double> Jac_output(vec_dim, vec_dim);
    dealii::Vector<double> Q_vector(vec_dim);
    ad_helper.register_dependent_variables(new_dofs);
    ad_helper.compute_values(Q_vector);
    ad_helper.compute_jacobian(Jac_output);

    // Multiply out to see if the two Jacobians are inverses
    dealii::FullMatrix<double> tmp(vec_dim, vec_dim);
    Jac_input.mmult(tmp, Jac_output);

    // Print everything out
    std::cout << "Printing product of Jacobians" << std::endl;
    tmp.print(std::cout, 10, 3);
    std::cout << std::endl << " Printing undiagonalized Q-vector"
              << Q_vector << std::endl;

    return 0;
}
