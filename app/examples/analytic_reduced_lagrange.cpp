#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/differentiation/ad.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

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

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    Q_vec[0] = 0.1;
    Q_vec[1] = 0.06;
    Q_vec[2] = 0.01;
    Q_vec[3] = 0.08;
    Q_vec[4] = 0.01;

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

    std::cout << R << std::endl;

    for (unsigned int i = 0; i < dim; ++i)
        std::cout << "\n" << eigs[i].first << "\n";

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

    std::vector<std::vector<dealii::FullMatrix<double>>>
        S_new(3, std::vector<dealii::FullMatrix<double>>(3, dealii::FullMatrix<double>(dim, msc::vec_dim<dim>)));
    for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
        {
            dB_ni_nj[0] = R[0][i] * R[0][j] - R[2][i] * R[2][j];
            dB_ni_nj[1] = R[0][i] * R[1][j] + R[1][i] * R[0][j];
            dB_ni_nj[2] = R[0][i] * R[2][j] + R[2][i] * R[0][j];
            dB_ni_nj[3] = R[1][i] * R[1][j] - R[2][i] * R[2][j];
            dB_ni_nj[4] = R[1][i] * R[2][j] + R[2][i] * R[1][j];

            for (unsigned int m = 0; m < msc::vec_dim<dim>; ++m)
                for (unsigned int n = 0; n < dim; ++n)
                {
                    S_new[i][j](n, m) = R[n][j] * dB_ni_nj[m];
                }
        }

    std::vector<dealii::FullMatrix<double>>
        dn(3, dealii::FullMatrix<double>(dim, msc::vec_dim<dim>));

    std::cout << "\n";
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            if (i != j)
            {
                dn[i].add(1 / (eigs[i].first - eigs[j].first), S_new[i][j]);
                S_new[i][j].print(std::cout, 15, 5);
                std::cout << "\n\n";

                std::cout << 1 / (eigs[i].first - eigs[j].first);
                std::cout << "\n\n";
            }


    std::vector<double> gamma(3);
    gamma[0] = 1 / (eigs[0].first - eigs[1].first);
    gamma[1] = 1 / (eigs[0].first - eigs[2].first);
    gamma[2] = 1 / (eigs[1].first - eigs[2].first);

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

    std::vector<dealii::FullMatrix<double>>
        T(2, dealii::FullMatrix<double>(msc::vec_dim<dim>, dim));

    for (unsigned int i = 0; i < 2; ++i)
    {
        T[i](0, 0) = 2*R[0][i];
        T[i](1, 0) = R[1][i];
        T[i](2, 0) = R[2][i];
        T[i](1, 1) = R[0][i];
        T[i](3, 1) = 2*R[1][i];
        T[i](4, 1) = R[2][i];
        T[i](2, 2) = R[0][i];
        T[i](4, 2) = R[1][i];
    }

    dealii::FullMatrix<double> dF(msc::vec_dim<dim>, 2);
    for (unsigned int i = 0; i < 2; ++i)
    {
        dF(0, i) = R[0][i]*R[0][i] - R[0][2]*R[0][2];
        dF(1, i) = R[0][i]*R[1][i] - R[0][2]*R[1][2];
        dF(2, i) = R[0][i]*R[2][i] - R[0][2]*R[2][2];
        dF(3, i) = R[1][i]*R[1][i] - R[1][2]*R[1][2];
        dF(4, i) = R[1][i]*R[2][i] - R[1][2]*R[2][2];
    }

    // S[0] *= (Lambda_red[0] - Lambda_red[1]) * gamma[0];
    // S[1] *= (2*Lambda_red[0] + Lambda_red[1]) * gamma[1];
    // S[2] *= (Lambda_red[0] + 2*Lambda_red[1]) * gamma[2];

    std::cout << "Printing T matrices\n";
    T[0].print(std::cout);
    std::cout << "\n";
    T[1].print(std::cout);
    std::cout << "\n";

    std::cout << "\nPrinting S matrices\n";
    S[0].print(std::cout);
    std::cout << "\n";
    S[1].print(std::cout);
    std::cout << "\n";
    S[2].print(std::cout);

    std::cout << "\nPrinting S_new matrices\n";
    S_new[0][1].print(std::cout);
    std::cout << "\n";
    S_new[0][2].print(std::cout);
    std::cout << "\n";
    S_new[1][2].print(std::cout);

    std::cout << "\nPrinting dF\n";
    dF.print(std::cout);

    std::cout << "\nPrinting dLambda\n";
    dLambda.print(std::cout);

    std::cout << "\nPrinting dlambda\n";
    dlambda.print(std::cout);
    std::cout << std::endl;

    std::cout << "\nPrinting gammas\n";
    std::cout << gamma[0] << "\n" << gamma[1] << "\n" << gamma[2] << "\n";

    std::vector<dealii::FullMatrix<double>>
        TS(3, dealii::FullMatrix<double>(msc::vec_dim<dim>, msc::vec_dim<dim>));
    T[0].mmult(TS[0], S[0]);
    T[0].mmult(TS[1], S[1]);
    T[1].mmult(TS[2], S[2]);


    dealii::FullMatrix<double> Jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    Jac.triple_product(dLambda, dF, dlambda);
    Jac.add((Lambda_red[0] - Lambda_red[1]) * gamma[0], TS[0],
            (2*Lambda_red[0] + Lambda_red[1]) * gamma[1], TS[1],
            (Lambda_red[0] + 2*Lambda_red[1]) * gamma[2], TS[2]);


    std::cout << "\nPrinting analytically calculated Jacobian\n\n";
    Jac.print(std::cout, 15, 5);
    std::cout << "\n\n" << std::endl;

    LagrangeMultiplier<3> lm(order, alpha, tol, max_iters);
    lm.invertQ(Q_vec);
    // lm.returnLambda(Lambda);

    dealii::LAPACKFullMatrix<double> lapack_jac;
    dealii::FullMatrix<double> regular_jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    lm.returnJac(lapack_jac);
    lapack_jac.invert();
    regular_jac = lapack_jac;

    std::cout << "\nPrinting regularly-calculated Jacobian\n\n";
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        for (unsigned int j = 0; j < msc::vec_dim<dim>; ++j)
            Jac(i, j) -= regular_jac(i, j);

    Jac.print(std::cout, 15, 5);


    const int output_dim = 11;
    // set names of things
    constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
        dealii::Differentiation::AD::NumberTypes::sacado_dfad;
    using ADHelper =
        dealii::Differentiation::AD::VectorFunction<dim, ADTypeCode, double>;
    using ADNumberType = typename ADHelper::ad_type;

    std::vector<double> Q_vector(msc::vec_dim<dim>);
    for (unsigned int i = 0; i < msc::vec_dim<dim>; i++)
        Q_vector[i] = Q_vec[i];

    // set up automatic differentiation
    ADHelper ad_helper(msc::vec_dim<dim>, output_dim);
    ad_helper.register_independent_variables(Q_vector);
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
    auto eigs_ad = dealii::eigenvectors(Q);

    dealii::Tensor<2, dim, ADNumberType> R_ad;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R_ad[i][j] = eigs_ad[j].second[i];

    // Need this to make sure it's a rotation matrix
    if (dealii::determinant(R_ad) < 0)
        R_ad *= -1;

    std::cout << std::endl
              << "Printing initial rotation matrix\n"
              << R_ad << std::endl
              << std::endl;

    std::vector<ADNumberType> outputs(output_dim);
    outputs[0] = eigs_ad[0].first;
    outputs[1] = eigs_ad[1].first;
    outputs[2] = R_ad[0][0];
    outputs[3] = R_ad[1][0];
    outputs[4] = R_ad[2][0];
    outputs[5] = R_ad[0][1];
    outputs[6] = R_ad[1][1];
    outputs[7] = R_ad[2][1];
    outputs[8] = R_ad[0][2];
    outputs[9] = R_ad[1][2];
    outputs[10] = R_ad[2][2];

    dealii::Vector<double> outputs_vec;
    dealii::FullMatrix<double> outputs_jac(output_dim, msc::vec_dim<dim>);

    ad_helper.register_dependent_variables(outputs);
    ad_helper.compute_values(outputs_vec);
    ad_helper.compute_jacobian(outputs_jac);

    std::cout << "\nPrinting auto-differentiated Jacobian:\n";
    outputs_jac.print(std::cout);

    for (unsigned int i = 0; i < dim; ++i) {
      std::cout << "\nPrinting dn_" + std::to_string(i) + ":\n\n";
      dn[i].print(std::cout);
    }

    ad_helper.reset(output_dim, msc::vec_dim<dim>);

    std::vector<double> outputs_vector(output_dim);
    for (unsigned int i = 2; i < output_dim; ++i)
        outputs_vector[i] = outputs_vec[i];

    outputs_vector[0] = Lambda_red[0];
    outputs_vector[1] = Lambda_red[1];

    ad_helper.register_independent_variables(outputs_vector);
    const std::vector<ADNumberType> output_ad
        = ad_helper.get_sensitive_variables();

    dealii::Tensor<2, dim, ADNumberType> R_ad_output;
    R_ad_output[0][0] = output_ad[2];
    R_ad_output[1][0] = output_ad[3];
    R_ad_output[2][0] = output_ad[4];
    R_ad_output[0][1] = output_ad[5];
    R_ad_output[1][1] = output_ad[6];
    R_ad_output[2][1] = output_ad[7];
    R_ad_output[0][2] = output_ad[8];
    R_ad_output[1][2] = output_ad[9];
    R_ad_output[2][2] = output_ad[10];

    dealii::SymmetricTensor<2, dim, ADNumberType> Lambda_mat_ad;
    Lambda_mat_ad[0][0] = output_ad[0];
    Lambda_mat_ad[1][1] = output_ad[1];
    Lambda_mat_ad[2][2] = -(output_ad[0] + output_ad[1]);

    dealii::Tensor<2, dim, ADNumberType> Lambda_mat_frame;
    Lambda_mat_frame = R_ad_output * Lambda_mat_ad * dealii::transpose(R_ad_output);


    std::vector<ADNumberType> final_outputs(msc::vec_dim<dim>);
    final_outputs[0] = Lambda_mat_frame[0][0];
    final_outputs[1] = Lambda_mat_frame[0][1];
    final_outputs[2] = Lambda_mat_frame[0][2];
    final_outputs[3] = Lambda_mat_frame[1][1];
    final_outputs[4] = Lambda_mat_frame[1][2];

    dealii::FullMatrix<double> final_output_jac(msc::vec_dim<dim>, output_dim);
    ad_helper.register_dependent_variables(final_outputs);
    ad_helper.compute_jacobian(final_output_jac);

    std::cout << "\n";
    final_output_jac.print(std::cout);

    dealii::FullMatrix<double> big_dLambda(output_dim, output_dim);
    for (unsigned int i = 0; i < output_dim; ++i)
        big_dLambda(i, i) = 1;

    big_dLambda(0, 0) = Jac_red[0][0];
    big_dLambda(1, 0) = Jac_red[1][0];
    big_dLambda(0, 1) = Jac_red[0][1];
    big_dLambda(1, 1) = Jac_red[1][1];

    dealii::FullMatrix<double>
        Jac_numerical(msc::vec_dim<dim>, msc::vec_dim<dim>);
    Jac_numerical.triple_product(big_dLambda, final_output_jac, outputs_jac);

    std::cout << "\nPrinting auto-diff singular Jacobian\n\n";
    Jac_numerical.print(std::cout, 10, 5);

    std::cout << "\nPrinting regular singular Jacobian\n\n";
    Jac.print(std::cout, 10, 5);

    std::cout << "L is size: (" << final_output_jac.m() << ", "
              << final_output_jac.n() << ")\n\n";
    std::cout << "J is size: (" << outputs_jac.m() << ", "
              << outputs_jac.n() << ")\n\n";
    std::cout << "K is size: (" << big_dLambda.m() << ", " << big_dLambda.n()
              << ")\n\n";

    return 0;
}
