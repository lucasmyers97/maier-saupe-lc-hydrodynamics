#ifndef LINEAR_INTERPOLATION
#define LINEAR_INTERPOLATION

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace{
    using mat =  std::vector<std::vector<double>>;
}

template <int dim>
class LinearInterpolationBase
{
protected:
    LinearInterpolationBase(){};
    LinearInterpolationBase(const std::vector<mat> Q_in, 
                            const mat X_in, const mat Y_in);

    std::vector<mat> Q_vec;
    mat X;
    mat Y;
};



template <int dim>
LinearInterpolationBase<dim>::LinearInterpolationBase
(const std::vector<mat> Q_in, const mat X_in, const mat Y_in)
 : Q_vec(Q_in)
 , X(X_in)
 , Y(Y_in)
{}



template <int dim>
class LinearInterpolation : public dealii::Function<dim>, 
                            protected LinearInterpolationBase<dim>
{
private:
    static constexpr int vec_dim = 5;

public:
    LinearInterpolation() : dealii::Function<dim>(vec_dim) {};
    LinearInterpolation(const std::vector<mat> Q_in, 
                        const mat X_in, const mat Y_in);

    void reinit(const std::vector<mat> Q_in, const mat X_in, const mat Y_in);

    virtual void vector_value_list
    (const std::vector<dealii::Point<dim>> &points,
     std::vector<dealii::Vector<double>> &values) const override;

    virtual void vector_gradient_list
        (const std::vector<dealii::Point<dim>> &points,
        std::vector<std::vector<dealii::Tensor<1, dim, double>>> &gradients)
        const override;
};



template <int dim>
LinearInterpolation<dim>::LinearInterpolation
(const std::vector<mat> Q_in, const mat X_in, const mat Y_in)
 : LinearInterpolationBase<dim>(Q_in, X_in, Y_in)
 , dealii::Function<dim>(LinearInterpolation::vec_dim)
{}



template <int dim>
void LinearInterpolation<dim>::reinit
(const std::vector<mat> Q_in, const mat X_in, const mat Y_in)
{
    this->Q_vec = Q_in;
    this->X = X_in;
    this->Y = Y_in;
}



template <int dim>
void LinearInterpolation<dim>::vector_value_list
(const std::vector<dealii::Point<dim>> &points, 
 std::vector<dealii::Vector<double>> &vals) const
{
    // since this is an equally-space grid, need to find extent + grid spacing
    int end_idx = this->X.size() - 1;
    double x_min = this->X[0][0];
    double x_max = this->X[end_idx][0];
    double x_spacing = (x_max - x_min) / (double(this->X.size()) - 1);

    double y_min = this->Y[0][0];
    double y_max = this->Y[0][end_idx];
    double y_spacing = (y_max - y_min) / (double(this->Y.size()) - 1);

    // for each point, find lower x- and y- coordinates in grid
    double x_dist = 0;
    double y_dist = 0;
    int x_crd = 0;
    int y_crd = 0;

    double x1 = 0;
    double y1 = 0;
    double x2 = 0;
    double y2 = 0;

    double x = 0;
    double y = 0;
    for (int i = 0; i < points.size(); ++i)
    {
        x = points[i][0];
        y = points[i][1];

        assert(x > x_min && x < x_max);
        assert(y > y_min && y < y_max);

        x_dist = x - x_min;
        y_dist = y - y_min;
        x_crd = std::floor(x_dist / x_spacing);
        y_crd = std::floor(y_dist / y_spacing);

        x1 = this->X[x_crd][0];
        y1 = this->Y[0][y_crd];
        x2 = this->X[x_crd + 1][0];
        y2 = this->Y[0][y_crd + 1];

        // interpolate Q's
        double Q11, Q12, Q21, Q22;
        double Qy1, Qy2;
        for (int j = 0; j < vec_dim; ++j)
        {
            // Q-values on vertices of cell
            Q11 = this->Q_vec[j][x_crd][y_crd];
            Q12 = this->Q_vec[j][x_crd][y_crd + 1];
            Q21 = this->Q_vec[j][x_crd + 1][y_crd];
            Q22 = this->Q_vec[j][x_crd + 1][y_crd + 1];

            // interpolate in x-direction
            Qy1 = (1 / x_spacing) * ( (x2 - x)*Q11 + (x - x1)*Q21 );
            Qy2 = (1 / x_spacing) * ( (x2 - x)*Q12 + (x - x1)*Q22 );

            vals[i][j] = (1 / y_spacing) * ( (y2 - y)*Qy1 - (y - y1)*Qy2 );
        }
    }
}



template <int dim>
void LinearInterpolation<dim>::vector_gradient_list
(const std::vector<dealii::Point<dim>> &points,
 std::vector<std::vector<dealii::Tensor<1, dim, double>>> &gradients) const
{}

#endif