#include "Utilities/LinearInterpolation.hpp"

#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <highfive/H5Easy.hpp>

#include <vector>
#include <string>
#include <utility>
#include <exception>

using mat = std::vector<std::vector<double>>;

std::pair<mat, mat> gen_meshgrid(int m, int n, double left, double right)
{
    mat X(m, std::vector<double>(n));
    mat Y(m, std::vector<double>(n));

    double dx = (right - left) / (m - 1);
    double dy = (right - left) / (n - 1);

    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
        {
            X[i][j] = dx*i + left;
            Y[i][j] = dy*j + left;
        }

    return std::make_pair(std::move(X), std::move(Y));
}




int main()
{
try {
    double left = -1;
    double right = 1;

    int m = 100;
    int n = 100;
    int vec_dim = 2;
    constexpr int space_dim = 2;

    std::string filename = "linear_interpolation_output.h5";

    // sample function on grid
    auto mesh_pair = gen_meshgrid(m, n, left, right);
    mat X = mesh_pair.first;
    mat Y = mesh_pair.second;
    std::vector<mat> f(vec_dim,
                       std::vector<std::vector<double>>(m,
                                                        std::vector<double>(n)));
    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
        {
            f[0][i][j] = X[i][j]*X[i][j];
            f[1][i][j] = Y[i][j]*Y[i][j];
        }

    // create linear interpolation object from sampled function
    LinearInterpolation<space_dim> lp(f, X, Y);

    // create points to evaluate linear interpolation at
    std::vector<dealii::Point<space_dim>> points(m*n);
    double scale = 0.995;
    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
        {
            points[i*n + j][0] = scale*X[i][j];
            points[i*n + j][1] = scale*Y[i][j];
        }

    // allocate vector and evaluate at points
    std::vector<dealii::Vector<double>>
        vals(m*n, dealii::Vector<double>(vec_dim));
    lp.vector_value_list(points, vals);

    // rewrite everything in a form that can be outputted
    std::vector<mat> interp_f(
        vec_dim, std::vector<std::vector<double>>(m, std::vector<double>(n)));
    mat interp_X(m, std::vector<double>(n));
    mat interp_Y(m, std::vector<double>(n));
    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
        {
            interp_f[0][i][j] = vals[i*n + j][0];
            interp_f[1][i][j] = vals[i*n + j][1];
            interp_X[i][j] = points[i*n + j][0];
            interp_Y[i][j] = points[i*n + j][1];
        }

    // print everything out so that we can plot it
    H5Easy::File file(filename, H5Easy::File::Overwrite);

    H5Easy::dump(file, "X", X);
    H5Easy::dump(file, "Y", Y);
    H5Easy::dump(file, "v1", f[0]);
    H5Easy::dump(file, "v2", f[1]);

    H5Easy::dump(file, "interp_X", interp_X);
    H5Easy::dump(file, "interp_Y", interp_Y);
    H5Easy::dump(file, "interp_v1", interp_f[0]);
    H5Easy::dump(file, "interp_v2", interp_f[1]);
}
catch (std::exception &exc)
{
    std::cerr << exc.what();
}

    return 0;
}
