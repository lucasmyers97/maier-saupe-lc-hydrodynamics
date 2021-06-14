#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "../extern-src/sphere_lebedev_rule.hpp"

using namespace Eigen;

// Have to put these here -- quirk of C++11
constexpr int LagrangeMultiplier::i[];
constexpr int LagrangeMultiplier::j[];
constexpr int LagrangeMultiplier::mat_dim;
constexpr int LagrangeMultiplier::order;

const Eigen::Map<Eigen::Matrix<double, LagrangeMultiplier::order, LagrangeMultiplier::mat_dim> >
LagrangeMultiplier::lebedev_coords = makeLebedevCoords();

const Eigen::Map<Eigen::Matrix<double, LagrangeMultiplier::order, 1> >
LagrangeMultiplier::lebedev_weights = makeLebedevWeights();

LagrangeMultiplier::LagrangeMultiplier(double in_alpha=1)
: alpha(in_alpha)
{
    assert(alpha <= 1);
}

Eigen::Map<Eigen::Matrix<double, LagrangeMultiplier::order, LagrangeMultiplier::mat_dim> >
LagrangeMultiplier::makeLebedevCoords()
{
    double *w = new double[order];
    double *coords = new double[3*order];

    ld_by_order(order, &coords[0], &coords[order], &coords[2*order], w);

    Map<Matrix<double, order, mat_dim> > coord_mat(coords);

    return coord_mat;
}

Eigen::Map<Eigen::Matrix<double, LagrangeMultiplier::order, 1> >
LagrangeMultiplier::makeLebedevWeights()
{
    double *w = new double[order];
    double *coords = new double[3*order];

    ld_by_order(order, &coords[0], &coords[order], &coords[2*order], w);

    Map<Matrix<double, order, 1> > weight_mat(w);

    return weight_mat;
}


double LagrangeMultiplier::sphereIntegral(
        std::function<double (double, double, double)> integrand)
{
    double *x;
	double *y;
	double *z;
	double *w;

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



void LagrangeMultiplier::Test(
        std::function<double (double, double, double)> f)
{
    Matrix<int, mat_dim, mat_dim> m;
    for (int k=0; k<vec_dim; ++k) {
        m(i[k], j[k]) = k;
        if (i[k] != j[k]) {
            m(j[k], i[k]) = k;
        }
    }

    std::cout << m << std::endl;
    std::cout << alpha << std::endl;

    double integral = sphereIntegral(f);

    std::cout << "Integral is:\n" << integral << std::endl;

    std::cout << lebedev_coords << std::endl;
    std::cout << lebedev_weights << std::endl;

}

