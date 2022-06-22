#ifndef PERIODIC_MU_2_STRESS_TENSOR_HPP
#define PERIODIC_MU_2_STRESS_TENSOR_HPP

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>

#include <cmath>
#include <cassert>

template <int dim>
class PeriodicMu2StressTensor : public dealii::TensorFunction<2, dim, double>
{
public:
    PeriodicMu2StressTensor(double A_, double B_,
                            double C_, double k_, double eps_)
        : dealii::TensorFunction<2, dim, double>()
        , A(A_)
        , B(B_)
        , C(C_)
        , k(k_)
        , eps(eps_)
    {};

    virtual dealii::Tensor<2, dim, double>
    value(const dealii::Point<dim> &p) const override;

    virtual void
    value_list(const std::vector<dealii::Point<dim>> &points,
               std::vector<dealii::Tensor<2, dim, double>> &values) const override;

private:

    double A;
    double B;
    double C;
    double k;
    double eps;
};



template <int dim>
dealii::Tensor<2, dim, double> PeriodicMu2StressTensor<dim>::
value(const dealii::Point<dim> &p) const
{
    dealii::Tensor<2, dim, double> H;

    H[0][0] = -A*(2.0/3.0) - B*(2.0/9.0) - C*(4.0/9.0);
    H[0][1] = -(1.0/3.0) * ( eps*(3*A + B + 2*C + 3*k*k) * std::sin(k*p[0]) );
    H[1][0] = H[0][1];
    H[1][1] = A*(1.0/3.0) + B*(1.0/9.0) + C*(2.0/9.0);

    if (dim == 3)
        H[2][2] = -(H[0][0] + H[1][1]);

    return H;
}



template <int dim>
void PeriodicMu2StressTensor<dim>::
value_list(const std::vector<dealii::Point<dim>> &points,
           std::vector<dealii::Tensor<2, dim, double>> &values) const
{
    assert((values.size() == points.size())
           && "Values and points different sizes");

    for (unsigned int i = 0; i < points.size(); ++i)
    {
      values[i][0][0] = -A * (2.0 / 3.0) - B * (2.0 / 9.0) - C * (4.0 / 9.0);
      values[i][1][0] = -(1.0/3.0) * ( eps*(3*A + B + 2*C + 3*k*k)
                                       * std::sin(k*points[i][0]) );
      values[i][0][1] = values[i][0][1];
      values[i][1][1] = A*(1.0/3.0) + B*(1.0/9.0) + C*(2.0/9.0);

      if (dim == 3)
          values[i][2][2] = -(values[i][0][0] + values[i][1][1]);
    }
}

#endif
