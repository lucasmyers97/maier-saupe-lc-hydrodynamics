#include "LagrangeMultiplier.hpp"
#include <cmath>
#include <iostream>
#include <deal.II/base/point.h>

int main()
{
    double alpha{1};
    double tol{1e-12};
    unsigned int max_iter{50};
    constexpr int order = 2702;

    LagrangeMultiplier<order> l(alpha, tol, max_iter);
    //l.lagrangeTest();

    return 0;
}
