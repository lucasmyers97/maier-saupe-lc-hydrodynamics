#include "BoundaryValues/DefectConfiguration.hpp"
#include "maier_saupe_constants.hpp"
#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <highfive/H5Easy.hpp>

int main()
{
    namespace msc = maier_saupe_constants;

    // set parameters for domain
    constexpr int dim = 3;
    int n_pts = 100;
    double l = 1.0;
    double dx = 2*l / (n_pts - 1);

    // construct meshgrid of deal.ii points
    std::vector<dealii::Point<dim>> point_list(n_pts*n_pts);
    double x = 0;
    double y = 0;
    for (int i = 0; i < n_pts; ++i)
    {
        x = i*dx - l;
        for (int j = 0; j < n_pts; ++j)
        {
            y = j*dx - l;
            point_list[i + n_pts*j][0] = x;
            point_list[i + n_pts*j][1] = y;
        }
    }

    // set up defect object
    double S = 1.0;
    double k = 0.5;
    DefectConfiguration<dim> defect_configuration;

    // allocate vector for Q-vector values, calculate defect config values
    std::vector<dealii::Vector<double>> 
        Q_list(n_pts*n_pts, dealii::Vector<double>(msc::vec_dim<dim>));
    defect_configuration.vector_value_list(point_list, Q_list);

    // for HighFive need matrices Q1,...,Q5 then X and Y (all n_pts x n_pts)
    using vec = std::vector<double>;
    using mat = std::vector<vec>;
    mat Q1(n_pts, vec(n_pts));
    mat Q2(n_pts, vec(n_pts));
    mat Q3(n_pts, vec(n_pts));
    mat Q4(n_pts, vec(n_pts));
    mat Q5(n_pts, vec(n_pts));

    mat X(n_pts, vec(n_pts));
    mat Y(n_pts, vec(n_pts));

    for (int i = 0; i < n_pts; ++i)
        for (int j = 0; j < n_pts; ++j)
        {
            Q1[i][j] = Q_list[i + j*n_pts][0];
            Q2[i][j] = Q_list[i + j*n_pts][1];
            Q3[i][j] = Q_list[i + j*n_pts][2];
            Q4[i][j] = Q_list[i + j*n_pts][3];
            Q5[i][j] = Q_list[i + j*n_pts][4];

            X[i][j] = point_list[i + j*n_pts][0];
            Y[i][j] = point_list[i + j*n_pts][1];
        }

    H5Easy::File file("defect_configuration.h5", H5Easy::File::Overwrite);
    H5Easy::dump(file, "Q1", Q1);
    H5Easy::dump(file, "Q2", Q2);
    H5Easy::dump(file, "Q3", Q3);
    H5Easy::dump(file, "Q4", Q4);
    H5Easy::dump(file, "Q5", Q5);

    H5Easy::dump(file, "X", X);
    H5Easy::dump(file, "Y", Y);

    return 0;
}
