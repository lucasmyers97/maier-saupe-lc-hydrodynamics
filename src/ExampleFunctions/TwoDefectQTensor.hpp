#ifndef TWO_DEFECT_Q_TENSOR_HPP
#define TWO_DEFECT_Q_TENSOR_HPP

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>

#include <cmath>
#include <vector>

template <int dim>
class TwoDefectQTensor : public dealii::TensorFunction<2, dim, double>
{
public:
    TwoDefectQTensor(std::vector<dealii::Point<dim>> centers_,
                     double r_)
        : dealii::TensorFunction<2, dim, double>()
        , centers(centers_)
        , r(r_)
    {};

    virtual dealii::Tensor<2, dim, double>
    value(const dealii::Point<dim> &p) const override;

    virtual void
    value_list(const std::vector<dealii::Point<dim>> &points,
               std::vector<dealii::Tensor<2, dim, double>> &values) const override;

private:

    std::vector<dealii::Point<dim>> centers;
    double r;
};



template <int dim>
dealii::Tensor<2, dim, double> TwoDefectQTensor<dim>::
value(const dealii::Point<dim> &p) const
{
    dealii::Tensor<2, dim, double> Q;

    double theta = 0;
    double m = 0;
    double S = 0;
    double r_coord = 0;
    for (unsigned int k = 0; k < centers.size(); ++k)
    {
        m = (k % 2) == 0 ? 1 : -1;
        theta += m * std::atan2(p[1] - centers[k][1], p[0] - centers[k][0]);
        r_coord = std::sqrt((p[0] - centers[k][0]) * (p[0] - centers[k][0])
                            + (p[1] - centers[k][1]) * (p[1] - centers[k][1]));
        S += 0.5 * (1 - std::exp(-r_coord / r));
    }

    Q[0][0] = 0.5 * S * ( (1.0/3.0) + std::cos(theta) );
    Q[1][0] = 0.5 * S * std::sin(theta);
    Q[0][1] = 0.5 * S * std::sin(theta);
    Q[1][1] = 0.5 * S * ( (1.0/3.0) - std::cos(theta) );

    if (dim == 3)
        Q[2][2] = 0.5 * S * (-1.0/3.0);

    return Q;
}



template <int dim>
void TwoDefectQTensor<dim>::
value_list(const std::vector<dealii::Point<dim>> &points,
           std::vector<dealii::Tensor<2, dim, double>> &values) const
{
    for (unsigned int i = 0; i < points.size(); ++i)
    {

        double theta = 0;
        double m = 0;
        double S = 0;
        double r_coord = 0;
        for (unsigned int k = 0; k < centers.size(); ++k)
        {
            m = (k % 2) == 0 ? 1 : -1;
            theta += m * std::atan2(points[i][1] - centers[k][1],
                                    points[i][0] - centers[k][0]);
            r_coord = std::sqrt((points[i][0] - centers[k][0])
                                * (points[i][0] - centers[k][0])
                                + (points[i][1] - centers[k][1])
                                * (points[i][1] - centers[k][1]));
            S += 0.5 * (1 - std::exp(-r_coord / r));

        }

        values[i][0][0] = 0.5 * S * ((1.0 / 3.0) + std::cos(theta));
        values[i][1][0] = 0.5 * S * std::sin(theta);
        values[i][0][1] = 0.5 * S * std::sin(theta);
        values[i][1][1] = 0.5 * S * ( (1.0/3.0) - std::cos(theta) );

        if (dim == 3)
            values[i][2][2] =  0.5 * S * (-1.0 / 3.0);
    }
}

#endif
