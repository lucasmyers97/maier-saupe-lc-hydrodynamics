#ifndef PLUS_HALF_Q_TENSOR_HPP
#define PLUS_HALF_Q_TENSOR_HPP

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>

#include <cmath>

template <int dim>
class PlusHalfQTensor : public dealii::TensorFunction<2, dim, double>
{
public:
    PlusHalfQTensor() : dealii::TensorFunction<2, dim, double>() {};

    virtual dealii::Tensor<2, dim, double>
    value(const dealii::Point<dim> &p) const override;

    virtual void
    value_list(const std::vector<dealii::Point<dim>> &points,
               std::vector<dealii::Tensor<2, dim, double>> &values) const override;
};



template <int dim>
dealii::Tensor<2, dim, double> PlusHalfQTensor<dim>::
value(const dealii::Point<dim> &p) const
{
    dealii::Tensor<2, dim, double> Q;

    double theta = std::atan2(p[1], p[0]);

    Q[0][0] = 0.5 * ((1.0/3.0) + std::cos(theta));
    Q[1][0] = 0.5 * std::sin(theta);
    Q[0][1] = 0.5 * std::sin(theta);
    Q[1][1] = 0.5 * ((1.0/3.0) - std::cos(theta));

    if (dim == 3)
        Q[2][2] = -(2.0/3.0);

    return Q;
}



template <int dim>
void PlusHalfQTensor<dim>::
value_list(const std::vector<dealii::Point<dim>> &points,
           std::vector<dealii::Tensor<2, dim, double>> &values) const
{
    for (unsigned int i = 0; i < points.size(); ++i)
    {
      double theta = std::atan2(points[i][1], points[i][0]);

      values[i][0][0] = 0.5 * ((1.0 / 3.0) + std::cos(theta));
      values[i][1][0] = 0.5 * std::sin(theta);
      values[i][0][1] = 0.5 * std::sin(theta);
      values[i][1][1] = 0.5 * ((1.0 / 3.0) - std::cos(theta));

      if (dim == 3)
        values[i][2][2] = -(2.0 / 3.0);
    }
}

#endif
