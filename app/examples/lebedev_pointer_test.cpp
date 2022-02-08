#include <iostream>
#include "sphere_lebedev_rule/sphere_lebedev_rule.hpp"

int main()
{
    const int order = 2702;
    double *x = new double[order];
    double *y = new double[order];
    double *z = new double[order];
    double *w = new double[order];

    double *coords = new double[4*order];

    ld_by_order(order, x, y, z, w);
    ld_by_order(order, &coords[0], &coords[order],
                &coords[2*order], &coords[3*order]);
   
    double sum = 0;
    for (int i=0; i<order; ++i) {
        sum += x[i] - coords[i];
        sum += y[i] - coords[order + i];
        sum += z[i] - coords[2*order + i];
        sum += w[i] - coords[3*order + i];
    }

    std::cout << sum << std::endl;

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;

    delete[] coords;

    return 0;
}
