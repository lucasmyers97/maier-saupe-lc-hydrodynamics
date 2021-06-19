#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/table_indices.h>
#include "sphere_lebedev_rule.hpp"

// Have to put these here -- quirk of C++11
constexpr int LagrangeMultiplier::vec_dim;
constexpr int LagrangeMultiplier::mat_dim;
constexpr int LagrangeMultiplier::order;
constexpr std::array<int, LagrangeMultiplier::vec_dim>
LagrangeMultiplier::i;
constexpr std::array<int, LagrangeMultiplier::vec_dim>
LagrangeMultiplier::j;

const std::vector<dealii::Point<LagrangeMultiplier::mat_dim>>
LagrangeMultiplier::lebedev_coords = makeLebedevCoords();
const std::vector<double>
LagrangeMultiplier::lebedev_weights = makeLebedevWeights();

std::vector<dealii::Point<LagrangeMultiplier::mat_dim>>
LagrangeMultiplier::makeLebedevCoords()
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    std::vector<dealii::Point<mat_dim>> coords;
    coords.reserve(order);
    for (int k = 0; k < order; ++k) {
        coords[k][0] = x[k];
        coords[k][1] = y[k];
        coords[k][2] = z[k];
    }

    delete x;
    delete y;
    delete z;
    delete w;

    return coords;
}

std::vector<double>
LagrangeMultiplier::makeLebedevWeights()
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    std::vector<double> weights;
    weights.reserve(order);
    for (int k = 0; k < order; ++k) {
        weights[k] = w[k];
    }

    delete x;
    delete y;
    delete z;
    delete w;

    return weights;
}

LagrangeMultiplier::LagrangeMultiplier(double in_alpha=1)
: alpha(in_alpha)
{
    assert(alpha <= 1);
}

double LagrangeMultiplier::sphereIntegral(
        std::function<double (double, double, double)> integrand)
{
    double *x, *y, *z, *w;

	x = new double[order];
	y = new double[order];
	z = new double[order];
	w = new double[order];
    
    ld_by_order(order, x, y, z, w);
    
    double integral;
    for (int k=0; k<order; ++k) {
        integral += w[k]*integrand(x[k], y[k], z[k]);
    }
    integral *= 4*M_PI;

    return integral;
}

void LagrangeMultiplier::printVecTest()
{
    double *x_ar, *y_ar, *z_ar, *w_ar;
    x_ar = new double[order];
    y_ar = new double[order];
    z_ar = new double[order];
    w_ar = new double[order];
    ld_by_order(order, x_ar, y_ar, z_ar, w_ar);
  
    int sum = 0;
    for (int k = 0; k < order; ++k) {
        sum += abs(lebedev_coords[k][0] - x_ar[k]);
        sum += abs(lebedev_coords[k][1] - y_ar[k]);
        sum += abs(lebedev_coords[k][2] - z_ar[k]);
        sum += abs(lebedev_weights[k] - w_ar[k]);
    }

    std::cout << "Sum is: " << sum << std::endl;
    
    delete x_ar;
    delete y_ar;
    delete z_ar;
    delete w_ar;
}

