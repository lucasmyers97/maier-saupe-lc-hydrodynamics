#include <iostream>
#include <cmath>
#include <vector>
#include <sphere_lebedev_rule.hpp>

double sin_theta(double x, double y, double z)
{
	// Return sin(theta) which will be integrated over the unit
	// sphere to give pi^2
	return sqrt(x*x + y*y);
}

double cos_theta(double x, double y, double z)
{
	// Return sin(theta) which will be integrated over the unit
	// sphere to give 0
	return z;
}

double one(double x, double y, double z)
{
	// Return 1, which will be integrated over the unit sphere
	// to give 4*pi
	return sqrt(x*x + y*y + z*z);
}

int main ()
{
	// get Lebedev quadrature weights
	const int order = 2702;
	double *x;
	double *y;
	double *z;
	double *w;

	x = new double[order];
	y = new double[order];
	z = new double[order];
	w = new double[order];

	ld_by_order(order, x, y, z, w);

	// integrate the above functions over the sphere
	double integral_1;
	double integral_2;
	double integral_3;
	for(int i=0; i<order; ++i) {
		integral_1 += w[i]*sin_theta(x[i], y[i], z[i]);
		integral_2 += w[i]*cos_theta(x[i], y[i], z[i]);
		integral_3 += w[i]*one(x[i], y[i], z[i]);
	}
	integral_1 *= 4*M_PI;
	integral_2 *= 4*M_PI;
	integral_3 *= 4*M_PI;

	// Print out results
	std::cout << "Lebedev gives: " << integral_1 << std::endl;
	std::cout << "Actual answer is: " << M_PI*M_PI << std::endl;
	std::cout << "Lebedev gives: " << integral_2 << std::endl;
	std::cout << "Actual answer is: " << 0 << std::endl;
	std::cout << "Lebedev gives: " << integral_3 << std::endl;
	std::cout << "Actual answer is: " << 4*M_PI << std::endl;



	return 0;
}
