#ifndef PERIODIC_Q_TENSOR_HPP
#define PERIODIC_Q_TENSOR_HPP

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>

#include <cmath>

template <int dim>
class PeriodicQTensor : public dealii::TensorFunction<2, dim, double>
{
public:
    PeriodicQTensor(double k_, double eps_)
        : dealii::TensorFunction<2, dim, double>()
        , k(k_)
        , eps(eps_)
    {};

    virtual dealii::Tensor<2, dim, double>
    value(const dealii::Point<dim> &p) const override;

    virtual void
    value_list(const std::vector<dealii::Point<dim>> &points,
               std::vector<dealii::Tensor<2, dim, double>> &values) const override;

private:

    double k;
    double eps;
};



template <int dim>
dealii::Tensor<2, dim, double> PeriodicQTensor<dim>::
value(const dealii::Point<dim> &p) const
{
    dealii::Tensor<2, dim, double> Q;

    Q[0][0] = 2.0/3.0;
    Q[0][1] = eps * std::sin(k*p[0]);
    Q[1][0] = Q[0][1];
    Q[1][1] = -1.0/3.0;

    if (dim == 3)
        Q[2][2] = -(Q[0][0] + Q[1][1]);

    return Q;
}



template <int dim>
void PeriodicQTensor<dim>::
value_list(const std::vector<dealii::Point<dim>> &points,
           std::vector<dealii::Tensor<2, dim, double>> &values) const
{
    for (unsigned int i = 0; i < points.size(); ++i)
    {
      values[i][0][0] = 2.0/3.0;
      values[i][1][0] = eps * std::sin(k * points[i][0]);
      values[i][0][1] = values[i][0][1];
      values[i][1][1] = -1.0/3.0;

      if (dim == 3)
          values[i][2][2] = -(values[i][0][0] + values[i][1][1]);
    }
}

#endif
