#include "LagrangeMultiplier.hpp"
#include <cmath>
#include <iostream>
#include <deal.II/base/point.h>

double f(dealii::Point<3> x)
{
    return sqrt(x[0]*x[0] + x[1]*x[1]);
}

int main()
{
    double alpha{0.5};
    LagrangeMultiplier l(alpha);

    l.printVecTest(f);

    dealii::Point<3> x{1.0, 5.0, 5.0};
    std::cout << x << std::endl;

    int m = 2;
    double y = x[m];
    auto numIntegrand = 
        [&l, y](dealii::Point<3> p)
        {return y*y * l.calcLagrangeExp(p);};
    std::cout << numIntegrand(x) << std::endl;

    // Note we've initialized Q and Lambda to 0
    // If you calculate the Q-value given a Lambda
    // which is all zeros, you again get zero (can
    // explicitly carry out the integrals), so the
    // residual should evaluate to zero
    l.updateRes();
    std::cout << l.Res << std::endl;

    return 0;
}
