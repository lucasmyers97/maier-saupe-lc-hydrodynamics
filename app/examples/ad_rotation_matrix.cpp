#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/differentiation/ad.h>
#include <deal.II/lac/vector.h>

#include <vector>
#include <iostream>

#include "Numerics/LagrangeMultiplier.hpp"
#include "Numerics/LagrangeMultiplierReduced.hpp"
#include "Utilities/maier_saupe_constants.hpp"
#include "Numerics/NumericalTools.hpp"

namespace msc = maier_saupe_constants;

int main()
{

    const int dim = 3;
    const int output_dim = 11;

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
    ADHelper ad_helper(msc::vec_dim<dim>, msc::vec_dim<dim>);
    // ADHelper ad_helper(msc::vec_dim<dim>, output_dim);
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

    auto q_pair = NumericalTools::matrix_to_quaternion(R_ad);

    // collect eigenvalues and rotation matrix entries into ad_outputs
    std::vector<ADNumberType> outputs(output_dim);
    outputs[0] = eigs_ad[0].first;
    outputs[1] = eigs_ad[1].first;
    outputs[2] = q_pair.first[0];
    outputs[3] = q_pair.first[1];
    outputs[4] = q_pair.first[2];
    // outputs[2] = R_ad[0][0];
    // outputs[3] = R_ad[1][0];
    // outputs[4] = R_ad[2][0];
    // outputs[5] = R_ad[0][1];
    // outputs[6] = R_ad[1][1];
    // outputs[7] = R_ad[2][1];
    // outputs[8] = R_ad[0][2];
    // outputs[9] = R_ad[1][2];
    // outputs[10] = R_ad[2][2];

    dealii::Vector<double> outputs_vec;
    // dealii::FullMatrix<double> outputs_jac(output_dim, msc::vec_dim<dim>);
    dealii::FullMatrix<double> outputs_jac(msc::vec_dim<dim>, msc::vec_dim<dim>);

    ad_helper.register_dependent_variables(outputs);
    ad_helper.compute_values(outputs_vec);
    ad_helper.compute_jacobian(outputs_jac);

    std::cout << "\nPrinting auto-differentiated Jacobian:\n";
    outputs_jac.print(std::cout);

    // calculate reduced Lagrange system without ad
    dealii::Tensor<1, 2, double> Q_red;
    Q_red[0] = outputs_vec[0];
    Q_red[1] = outputs_vec[1];

    dealii::Tensor<1, 2, double> Lambda_red;
    dealii::Tensor<2, 2, double> Jac_red;
    LagrangeMultiplierReduced lmr(order, alpha, tol, max_iters);
    lmr.invertQ(Q_red);
    Lambda_red = lmr.returnLambda();
    Jac_red = lmr.returnJac();

    // redo ad calculation for return transformation
    // ad_helper.reset(output_dim, msc::vec_dim<dim>);
    ad_helper.reset(msc::vec_dim<dim>, msc::vec_dim<dim>);

    // leave rotation matrix entries unchanged
    // std::vector<double> outputs_vector(output_dim);
    std::vector<double> outputs_vector(msc::vec_dim<dim>);
    // for (unsigned int i = 2; i < output_dim; ++i)
    //     outputs_vector[i] = outputs_vec[i];

    // update 2 other dofs to be lambda eigenvalues
    outputs_vector[0] = Lambda_red[0];
    outputs_vector[1] = Lambda_red[1];
    outputs_vector[2] = outputs_vec[2];
    outputs_vector[3] = outputs_vec[3];
    outputs_vector[4] = outputs_vec[4];

    ad_helper.register_independent_variables(outputs_vector);
    const std::vector<ADNumberType> output_ad
        = ad_helper.get_sensitive_variables();

    q_pair.first[0] = output_ad[2];
    q_pair.first[1] = output_ad[3];
    q_pair.first[2] = output_ad[4];
    auto R_ad_output = NumericalTools::quaternion_to_matrix(q_pair.first,
                                                            q_pair.second);
    // dealii::Tensor<2, dim, ADNumberType> R_ad_output;
    // R_ad_output[0][0] = output_ad[2];
    // R_ad_output[1][0] = output_ad[3];
    // R_ad_output[2][0] = output_ad[4];
    // R_ad_output[0][1] = output_ad[5];
    // R_ad_output[1][1] = output_ad[6];
    // R_ad_output[2][1] = output_ad[7];
    // R_ad_output[0][2] = output_ad[8];
    // R_ad_output[1][2] = output_ad[9];
    // R_ad_output[2][2] = output_ad[10];

    dealii::SymmetricTensor<2, dim, ADNumberType> Lambda_mat_ad;
    Lambda_mat_ad[0][0] = output_ad[0];
    Lambda_mat_ad[1][1] = output_ad[1];
    Lambda_mat_ad[2][2] = -(output_ad[0] + output_ad[1]);

    dealii::Tensor<2, dim, ADNumberType> Lambda_mat_frame;
    Lambda_mat_frame = R_ad_output * Lambda_mat_ad * dealii::transpose(R_ad_output);


    // collect final outputs into 5-component vector
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

    // calculate derivative of middle transformation which leaves rotation
    // matrix intact
    // dealii::FullMatrix<double> big_dLambda(output_dim, output_dim);
    // for (unsigned int i = 0; i < output_dim; ++i)
    //     big_dLambda(i, i) = 1;
    dealii::FullMatrix<double> big_dLambda(msc::vec_dim<dim>, msc::vec_dim<dim>);
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
      big_dLambda(i, i) = 1;

    Jac_red = dealii::invert(Jac_red);
    big_dLambda(0, 0) = Jac_red[0][0];
    big_dLambda(1, 0) = Jac_red[1][0];
    big_dLambda(0, 1) = Jac_red[0][1];
    big_dLambda(1, 1) = Jac_red[1][1];

    dealii::FullMatrix<double>
        Jac_numerical(msc::vec_dim<dim>, msc::vec_dim<dim>);
    Jac_numerical.triple_product(big_dLambda, final_output_jac, outputs_jac);

    std::cout << "\nPrinting auto-diff singular Jacobian\n\n";
    Jac_numerical.print(std::cout, 10, 5);

    LagrangeMultiplier<3> lm(order, alpha, tol, max_iters);
    lm.invertQ(Q_vec);
    // lm.returnLambda(Lambda);
    dealii::FullMatrix<double> Jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    dealii::LAPACKFullMatrix<double> lapack_jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    lm.returnJac(lapack_jac);
    lapack_jac.invert();
    Jac = lapack_jac;
    // Jac.print(std::cout, 10);

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
