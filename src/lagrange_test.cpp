#include "LagrangeMultiplier.h"
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
    l.Test();

    int order = 2702; 
    double integral = l.sphereIntegral(order, f);

    std::cout << "Integral is:\n" << integral << std::endl;

    return 0;
}
