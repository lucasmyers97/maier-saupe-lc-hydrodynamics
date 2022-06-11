#ifndef TENSOR_TO_VECTOR_HPP
#define TENSOR_TO_VECTOR_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>

#include <cmath>
#include <vector>
#include <memory>
#include <utility>

#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

template <int dim, typename TensorFunction>
class TensorToVector : public dealii::Function<dim>
{
public:

    template <typename... Args>
    TensorToVector(Args&&... args)
        : dealii::Function<dim>(msc::vec_dim<dim>)
        , tensor_function(std::make_unique<TensorFunction>(std::forward<Args>(args)...))
    {};

    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &values) const override;

    virtual void
    vector_value_list(const std::vector<dealii::Point<dim>> &points,
                      std::vector<dealii::Vector<double>> &values) const override;

private:

    std::unique_ptr<dealii::TensorFunction<2, dim, double>> tensor_function;
};



template <int dim, typename TensorFunction>
void TensorToVector<dim, TensorFunction>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double> &Q_vec) const
{
    dealii::Tensor<2, dim, double> Q;
    Q = tensor_function->value(p);

    Q_vec[0] = Q[0][0];
    Q_vec[1] = Q[0][1];
    Q_vec[3] = Q[1][1];

    if (dim == 3)
    {
        Q_vec[2] = Q[0][2];
        Q_vec[4] = Q[1][2];
    }
}



template <int dim, typename TensorFunction>
void TensorToVector<dim, TensorFunction>::
vector_value_list(const std::vector<dealii::Point<dim>> &points,
                  std::vector<dealii::Vector<double>> &values) const
{
    std::vector<dealii::Tensor<2, dim, double>> Q_vals(points.size());
    tensor_function->value_list(points, Q_vals);

    for (unsigned int i = 0; i < points.size(); i++)
    {
        values[i][0] = Q_vals[i][0][0];
        values[i][1] = Q_vals[i][0][1];
        values[i][3] = Q_vals[i][1][1];

        if (dim == 3)
        {
            values[i][2] = Q_vals[i][0][2];
            values[i][4] = Q_vals[i][1][2];
        }
    }
}

#endif
