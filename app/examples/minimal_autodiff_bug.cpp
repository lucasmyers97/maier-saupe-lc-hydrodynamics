#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/differentiation/ad.h>

#include <vector>
#include <array>
#include <utility>
#include <iostream>

int main()
{
    const int dim = 3;
    const int vec_dim = 5;

    std::vector<double> Q_vec({-0.125,
                               -0.25,
                               0,
                               0.35,
                               0});

    // set names of things
    constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
        dealii::Differentiation::AD::NumberTypes::sacado_dfad;
    using ADHelper =
        dealii::Differentiation::AD::VectorFunction<dim, ADTypeCode, double>;
    using ADNumberType = typename ADHelper::ad_type;

    // set up automatic differentiation
    ADHelper ad_helper(vec_dim, vec_dim);
    ad_helper.register_independent_variables(Q_vec);
    const std::vector<ADNumberType> Q_ad = ad_helper.get_sensitive_variables();

    // diagonalize and keep track of eigen-numbers
    dealii::SymmetricTensor<2, dim, ADNumberType> Q;
    Q[0][0] = Q_ad[0];
    Q[0][1] = Q_ad[1];
    Q[0][2] = Q_ad[2];
    Q[1][1] = Q_ad[3];
    Q[1][2] = Q_ad[4];
    Q[2][2] = -(Q_ad[0] + Q_ad[3]);

    auto eigs = dealii::eigenvectors(Q,
                                     dealii::SymmetricTensorEigenvectorMethod::jacobi);
                                     // dealii::SymmetricTensorEigenvectorMethod::ql_implicit_shifts);

    dealii::Tensor<2, dim, ADNumberType> R;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R[i][j] = eigs[j].second[i];

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
          std::cout << R[i][j] << std::endl;

    return 0;
}
