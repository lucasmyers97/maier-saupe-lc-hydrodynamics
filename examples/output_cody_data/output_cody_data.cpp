#include <highfive/H5DataSet.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <cmath>

constexpr int vec_dim = 5;
using namespace dealii;
using mat =  std::vector<std::vector<double>>;

template <int dim>
class SolutionBase
{
protected:
    SolutionBase(const std::vector<mat> Q_in, const mat X_in, const mat Y_in);

    const std::vector<mat> Q_vec;
    const mat X;
    const mat Y;
};



template <int dim>
SolutionBase<dim>::SolutionBase
(const std::vector<mat> Q_in, const mat X_in, const mat Y_in)
 : Q_vec(Q_in)
 , X(X_in)
 , Y(Y_in)
{}



template <int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
public:
    Solution(const std::vector<mat> Q_in, const mat X_in, const mat Y_in);

    virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                   std::vector<Vector<double>> &values) 
                                   const override;

    virtual void vector_gradient_list
        (const std::vector<Point<dim>> &points,
        std::vector<std::vector<Tensor<1, dim, double>>> &gradients)
        const override;
};



template <int dim>
Solution<dim>::Solution
(const std::vector<mat> Q_in, const mat X_in, const mat Y_in)
 : SolutionBase<dim>(Q_in, X_in, Y_in)
{}



template <int dim>
void Solution<dim>::vector_value_list(const std::vector<Point<dim>> &points,
                                      std::vector<Vector<double>> &vals) const
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
void Solution<dim>::vector_gradient_list
(const std::vector<Point<dim>> &points,
 std::vector<std::vector<Tensor<1, dim, double>>> &gradients) const
{}



int main()
{
    std::string filename = DATA_FILE_LOCATION;
    
    HighFive::File f(filename);
    size_t num_objects = f.getNumberObjects();
    std::vector<std::string> object_names = f.listObjectNames();

    for (int i = 0; i < num_objects; ++i)
        std::cout << object_names[i] << std::endl;

    std::cout << std::endl;

    using mat =  std::vector<std::vector<double>>;
    std::vector<mat> Q_vec(vec_dim);
    mat X;
    mat Y;

    std::vector<HighFive::DataSet> dset(num_objects);

    for (int i = 0; i < vec_dim; ++i)
    {
        dset[i] = f.getDataSet(object_names[i]);
        dset[i].read(Q_vec[i]);
    }

    dset[num_objects - 2] = f.getDataSet(object_names[num_objects - 2]);
    dset[num_objects - 2].read(X);
    dset[num_objects - 1] = f.getDataSet(object_names[num_objects - 1]);
    dset[num_objects - 1].read(Y);

    constexpr int dim = 2;
    Solution<dim> sol(Q_vec, X, Y);

    int num_vals = 1;
    std::vector<Vector<double>> vals(num_vals, Vector<double>(vec_dim));
    std::vector<Point<dim>> p(num_vals, Point<dim>(5.5, 3.14));

    sol.vector_value_list(p, vals);
    std::cout << vals[0] << std::endl;

    return 0;
}