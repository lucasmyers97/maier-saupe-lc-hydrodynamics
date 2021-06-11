#include "LagrangeMultiplier.hpp"
#include <cmath>
#include <iostream>

double f(double x, double y, double z)
{
    return sqrt(x*x + y*y);
}

int main()
{
    double alpha{0.5};
    LagrangeMultiplier l(alpha);

    int order = 2702;
    l.Test(order, f);

    return 0;
}
