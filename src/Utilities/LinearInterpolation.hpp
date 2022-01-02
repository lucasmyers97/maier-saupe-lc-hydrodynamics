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
#include <stdexcept>



template <int dim>
class LinearInterpolation : public dealii::Function<dim>
{
    using mat = std::vector<std::vector<double>>;

public:
    LinearInterpolation(const unsigned int n_components=1);
    LinearInterpolation(const std::vector<mat> f_in, 
                        const mat X_in, const mat Y_in);

    void reinit(const std::vector<mat> Q_in, const mat X_in, const mat Y_in);

    virtual void vector_value_list
    (const std::vector<dealii::Point<dim>> &points,
     std::vector<dealii::Vector<double>> &values) const override;

    virtual void vector_gradient_list
        (const std::vector<dealii::Point<dim>> &points,
        std::vector<std::vector<dealii::Tensor<1, dim, double>>> &gradients)
        const override;

private:
    /* \brief vector-function evaluated at grid-points */
    std::vector<mat> f;
    /* \brief vector dimension of f */
    unsigned int vec_dim;

    /* \brief grid dimension in x-direction */
    unsigned int m;
    /* \brief grid dimension in y-direction */
    unsigned int n;
    /* \brief smallest grid-value in x-direction */
    double x_start;
    /* \brief smallest grid-value in y-direction */
    double y_start;
    /* \brief grid-spacing in the x-direction */
    double dx;
    /* \brief grid-spacing in the y-direction */
    double dy;
};



template <int dim>
LinearInterpolation<dim>::LinearInterpolation(const unsigned int n_components)
    : dealii::Function<dim>(n_components)
{}



template <int dim>
LinearInterpolation<dim>::LinearInterpolation(const std::vector<mat> f_,
                                              const mat X, const mat Y)
    : dealii::Function<dim>(LinearInterpolation::vec_dim)
    , f(f_)
{
    vec_dim = f.size();

    m = X.size();
    n = X[0].size();
    if (m != Y.size() || n != Y[0].size())
    {
      throw std::runtime_error(std::string(
                "X- and Y- grids not the same size!"));
      return;
    }
    if (m == 0 || n == 0)
    {
      throw std::runtime_error(std::string(
                "One grid-dimension is empty"));
      return;
    }

    x_start = X[0][0];
    y_start = Y[0][0];

    dx = (X.back()[0] - X[0][0]) / (m - 1);
    dy = (Y[0].back() - Y[0][0]) / (n - 1);
    if (dx <= 0 || dy <= 0)
    {
      throw std::runtime_error(std::string(
                "Grids are incorrectly ordered"));
      return;
    }
}



template <int dim>
void LinearInterpolation<dim>::reinit
(const std::vector<mat> f_, const mat X, const mat Y)
{
    f = f_;
    vec_dim = f.size();

    m = X.size();
    n = X[0].size();
    if (m != Y.size() || n != Y[0].size())
    {
      throw std::runtime_error(std::string(
                "X- and Y- grids not the same size!"));
      return;
    }
    if (m == 0 || n == 0)
    {
      throw std::runtime_error(std::string(
                "One grid-dimension is empty"));
      return;
    }

    x_start = X[0][0];
    y_start = Y[0][0];

    dx = (X.back()[0] - X[0][0]) / (m - 1);
    dy = (Y[0].back() - Y[0][0]) / (n - 1);
    if (dx <= 0 || dy <= 0)
    {
      throw std::runtime_error(std::string(
                "Grids are incorrectly ordered"));
      return;
    }
}



template <int dim>
void LinearInterpolation<dim>::vector_value_list(
    const std::vector<dealii::Point<dim>> &points,
    std::vector<dealii::Vector<double>> &vals) const
{
    // check sizing
    assert(points.size() == vals.size()
           && "Number of points and values is different");
    assert(vals[0].size() == f.size()
           && "Input vector and sampled vector are different sizes");

    // find lower-left indices associated with each point
    // allocate n_pts x dim indices
    std::vector<std::vector<int>> idx(points.size(),
                                      std::vector<int>(dim));
    for (unsigned int pt_idx = 0; pt_idx < points.size(); ++pt_idx)
    {
        double num_x_gridpoints = (points[pt_idx][0] - x_start) / dx;
        double num_y_gridpoints = (points[pt_idx][1] - y_start) / dy;
        idx[pt_idx][0] = static_cast<unsigned int>( std::floor(num_x_gridpoints) );
        idx[pt_idx][1] = static_cast<unsigned int>( std::floor(num_y_gridpoints) );
    }

    // do the interpolation -- for an explanation of the coefficients, see wiki
    for (unsigned int pt_idx = 0; pt_idx < points.size(); ++pt_idx)
    {
        // lower left grid indices corresponding to given point
        unsigned int i = idx[pt_idx][0];
        unsigned int j = idx[pt_idx][1];

        if (i < 0 || i >= (m - 1) || j < 0 || j >= (n - 1))
        {
            throw std::runtime_error(std::string(
                      "Error: Interpolation point is out of bounds\n"));
            return;
        }

        double x = points[pt_idx][0];
        double y = points[pt_idx][1];
        double x1 = dx * i + x_start;
        double x2 = dx * (i + 1) + x_start;
        double y1 = dy * j + y_start;
        double y2 = dy * (j + 1) + y_start;

        // coefficients of final interpolation sum
        double c11 = (x2 - x)*(y2 - y);
        double c21 = (x - x1)*(y2 - y);
        double c12 = (x2 - x)*(y - y1);
        double c22 = (x - x1)*(y - y1);
        double A = 1 / (dx*dy);

        for (unsigned int vec_idx = 0; vec_idx < vec_dim; ++vec_idx)
        {
            double f11 = f[vec_idx] [i][j];
            double f21 = f[vec_idx] [i + 1][j];
            double f12 = f[vec_idx] [i][j + 1];
            double f22 = f[vec_idx] [i + 1][j + 1];
            vals[pt_idx][vec_idx] = A * (f11*c11 + f21*c21 + f12*c12 + f22*c22);
        }
    }
}



// template <int dim>
// void LinearInterpolation<dim>::vector_value_list
// (const std::vector<dealii::Point<dim>> &points, 
//  std::vector<dealii::Vector<double>> &vals) const
// {
//     // since this is an equally-space grid, need to find extent + grid spacing
//     int end_idx = this->X.size() - 1;
//     double x_min = this->X[0][0];
//     double x_max = this->X[end_idx][0];
//     double x_spacing = (x_max - x_min) / (double(this->X.size()) - 1);

//     double y_min = this->Y[0][0];
//     double y_max = this->Y[0][end_idx];
//     double y_spacing = (y_max - y_min) / (double(this->Y.size()) - 1);

//     // for each point, find lower x- and y- coordinates in grid
//     double x_dist = 0;
//     double y_dist = 0;
//     int x_crd = 0;
//     int y_crd = 0;

//     double x1 = 0;
//     double y1 = 0;
//     double x2 = 0;
//     double y2 = 0;

//     double x = 0;
//     double y = 0;
//     for (int i = 0; i < points.size(); ++i)
//     {
//         x = points[i][0];
//         y = points[i][1];

//         assert(x > x_min && x < x_max);
//         assert(y > y_min && y < y_max);

//         x_dist = x - x_min;
//         y_dist = y - y_min;
//         x_crd = std::floor(x_dist / x_spacing);
//         y_crd = std::floor(y_dist / y_spacing);

//         x1 = this->X[x_crd][0];
//         y1 = this->Y[0][y_crd];
//         x2 = this->X[x_crd + 1][0];
//         y2 = this->Y[0][y_crd + 1];

//         // interpolate Q's
//         double Q11, Q12, Q21, Q22;
//         double Qy1, Qy2;
//         for (int j = 0; j < vec_dim; ++j)
//         {
//             // Q-values on vertices of cell
//             Q11 = this->Q_vec[j][x_crd][y_crd];
//             Q12 = this->Q_vec[j][x_crd][y_crd + 1];
//             Q21 = this->Q_vec[j][x_crd + 1][y_crd];
//             Q22 = this->Q_vec[j][x_crd + 1][y_crd + 1];

//             // interpolate in x-direction
//             Qy1 = (1 / x_spacing) * ( (x2 - x)*Q11 + (x - x1)*Q21 );
//             Qy2 = (1 / x_spacing) * ( (x2 - x)*Q12 + (x - x1)*Q22 );

//             vals[i][j] = (1 / y_spacing) * ( (y2 - y)*Qy1 - (y - y1)*Qy2 );
//         }
//     }
// }



template <int dim>
void LinearInterpolation<dim>::vector_gradient_list
(const std::vector<dealii::Point<dim>> &points,
 std::vector<std::vector<dealii::Tensor<1, dim, double>>> &gradients) const
{}

#endif
