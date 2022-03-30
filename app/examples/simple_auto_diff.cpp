#include <deal.II/differentiation/ad.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>

#include <vector>
#include <cmath>
#include <iostream>

template <typename NumberType>
std::vector<NumberType> func(const std::vector<NumberType> &x)
{
    std::vector<NumberType> output(2);
    output[0] = std::cos(x[0] / x[1]);
    output[1] = std::sin(x[0] / x[1]);

    return output;
}

int main()
{
    const int dim = 3;

    constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
        dealii::Differentiation::AD::NumberTypes::sacado_dfad;
    using ADHelper =
        dealii::Differentiation::AD::VectorFunction<dim, ADTypeCode, double>;
    using ADNumberType = typename ADHelper::ad_type;

    ADHelper ad_helper(2, 2);
    std::vector<double> vec({0.5, 2});
    ad_helper.register_independent_variables(vec);
    const std::vector<ADNumberType> independent_variables_ad
        = ad_helper.get_sensitive_variables();
    const std::vector<ADNumberType> output = func(independent_variables_ad);

    ad_helper.register_dependent_variables(output);

    dealii::FullMatrix<double> Jac(2, 2);
    ad_helper.compute_jacobian(Jac);

    Jac.print(std::cout);

    return 0;
}
