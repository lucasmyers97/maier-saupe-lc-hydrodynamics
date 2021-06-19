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

    return 0;
}
