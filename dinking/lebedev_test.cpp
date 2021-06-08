#include <iostream>
#include <cmath>
#include <vector>
#include "../extern-src/sphere_lebedev_rule.hpp"

int main ()
{
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

	// integrate z = sin(theta) over the sphere
	// should get pi^2
	double integral;
	for(int i=0; i<order; ++i) {
		integral += w[i]*sqrt(x[i]*x[i] + y[i]*y[i]);
	}
	integral *= 4*M_PI;
	std::cout << "Lebedev gives: " << integral << std::endl;
	std::cout << "Actual answer is: " << M_PI*M_PI << std::endl;
	return 0;
}
