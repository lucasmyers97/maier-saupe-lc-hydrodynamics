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
LagrangeMultiplier::Q_row;
constexpr std::array<int, LagrangeMultiplier::vec_dim>
LagrangeMultiplier::Q_col;
constexpr std::array<
    std::array<int, LagrangeMultiplier::mat_dim>, 
    LagrangeMultiplier::mat_dim> 
LagrangeMultiplier::Q_idx;

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
    Lambda.reinit(vec_dim);
}

double LagrangeMultiplier::sphereIntegral(
        std::function<double 
        (dealii::Point<LagrangeMultiplier::mat_dim>)> integrand)
{
    double integral;
    for (int k=0; k<order; ++k) {
        integral += lebedev_weights[k]*integrand(lebedev_coords[k]);
    }
    integral *= 4*M_PI;

    return integral;
}

double LagrangeMultiplier::numIntegrand(
        dealii::Point<LagrangeMultiplier::mat_dim> x,
        int m)
{
    double sum = 0;
    for (int k = 0; k < mat_dim; ++k) {
        for (int l = 0; l < mat_dim; ++l) {
            sum += Lambda[Q_idx[k][l]]*x[k]*x[l];
        }
    }
    // Have to correct for last entry being sum
    // of two entries
    sum -= (2*Lambda[0] + Lambda[3]);

    return x[m]*x[m]*exp(sum);
}

void LagrangeMultiplier::updateRes()
{
    
}

void LagrangeMultiplier::printVecTest(
        std::function<double 
        (dealii::Point<LagrangeMultiplier::mat_dim>)> f)
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];
    ld_by_order(order, x, y, z, w);
  
    int sum = 0;
    for (int k = 0; k < order; ++k) {
        sum += abs(lebedev_coords[k][0] - x[k]);
        sum += abs(lebedev_coords[k][1] - y[k]);
        sum += abs(lebedev_coords[k][2] - z[k]);
        sum += abs(lebedev_weights[k] - w[k]);
    }

    std::cout << "Sum is: " << sum << std::endl; 
    delete x;
    delete y;
    delete z;
    delete w;

    double integral = sphereIntegral(f);
    std::cout << "Integral is: " << integral << std::endl;
    std::cout << "Integral is supposed to be: "
        << M_PI*M_PI << std::endl;
}

