#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/table_indices.h>
#include "sphere_lebedev_rule.hpp"

// Have to put these here -- quirk of C++11
constexpr int LagrangeMultiplier::i[];
constexpr int LagrangeMultiplier::j[];
constexpr int LagrangeMultiplier::mat_dim;
constexpr int LagrangeMultiplier::order;

const dealii::Vector<double>
LagrangeMultiplier::x = makeLebedev('x');
const dealii::Vector<double>
LagrangeMultiplier::y = makeLebedev('y');
const dealii::Vector<double>
LagrangeMultiplier::z = makeLebedev('z');
const dealii::Vector<double>
LagrangeMultiplier::w = makeLebedev('w');

dealii::Vector<double>
LagrangeMultiplier::makeLebedev(char c)
{
    double *x_ar, *y_ar, *z_ar, *w_ar;
    x_ar = new double[order];
    y_ar = new double[order];
    z_ar = new double[order];
    w_ar = new double[order];

    ld_by_order(order, x_ar, y_ar, z_ar, w_ar);

    double *coord_ar;
    
    switch (c) {
        case 'x':
            coord_ar = x_ar;
            break;
        case 'y':
            coord_ar = y_ar;
            break;
        case 'z':
            coord_ar = z_ar;
            break;
        case 'w':
            coord_ar = w_ar;
            break;
    }

    dealii::Vector<double> coord(coord_ar, coord_ar + order);
    
    return coord;
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
        sum += abs(x[k] - x_ar[k]);
        sum += abs(y[k] - y_ar[k]);
        sum += abs(z[k] - z_ar[k]);
        sum += abs(w[k] - w_ar[k]);
    }

    std::cout << "Sum is: " << sum << std::endl;
    
    delete x_ar;
    delete y_ar;
    delete z_ar;
    delete w_ar;
}

