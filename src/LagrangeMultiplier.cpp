#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "../extern-src/sphere_lebedev_rule.hpp"

using namespace Eigen;

LagrangeMultiplier::LagrangeMultiplier(double in_alpha=1)
: alpha(in_alpha)
{
    assert(alpha <= 1);

    if (vec_dim == 5) {
        i(0) = 0;
        i(1) = 0;
        i(2) = 0;
        i(3) = 1;
        i(4) = 1;

        j(0) = 0;
        j(1) = 1;
        j(2) = 2;
        j(3) = 1;
        j(4) = 2;
    }
}

double LagrangeMultiplier::sphereIntegral(int order,
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
    for (int i=0; i<order; ++i) {
        integral += w[i]*integrand(x[i], y[i], z[i]);
    }
    integral *= 4*M_PI;

    return integral;
}



void LagrangeMultiplier::Test(int order, 
        std::function<double (double, double, double)> f)
{
    Matrix<int, mat_dim, mat_dim> m;
    for (int k=0; k<vec_dim; ++k) {
        m(i(k), j(k)) = k;
        if (i(k) != j(k)) {
            m(j(k), i(k)) = k;
        }
    }

    std::cout << m << std::endl;
    std::cout << alpha << std::endl;

    double integral = sphereIntegral(order, f);

    std::cout << "Integral is:\n" << integral << std::endl;

}
