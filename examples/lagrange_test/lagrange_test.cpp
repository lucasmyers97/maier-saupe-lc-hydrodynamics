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
    double alpha{1};
    double tol{1e-12};
    unsigned int max_iter{50};
    LagrangeMultiplier l(alpha, tol, max_iter);

    l.printVecTest(f);

    dealii::Point<3> x{1.0, 5.0, 5.0};
    std::cout << x << std::endl;

    int m = 2;
    double y = x[m];
    auto numIntegrand = 
        [&l, y](dealii::Point<3> p)
        {return y*y * l.lambdaSum(p);};
    std::cout << numIntegrand(x) << std::endl;

    dealii::Vector<double> new_Q({2.0/4.0 - 1e-2,0.0,0.0,-1.0/4.0,0.0});
    l.setQ(new_Q);

    // Note we've initialized Q and Lambda to 0
    // If you calculate the Q-value given a Lambda
    // which is all zeros, you again get zero (can
    // explicitly carry out the integrals), so the
    // residual should evaluate to zero
    l.updateRes();
    std::cout << l.Res << std::endl;
    std::cout << std::endl;

    l.updateJac();
    l.Jac.print_formatted(std::cout);

    l.updateVariation();
    std::cout << l.dLambda << std::endl;

    l.updateJac();
//    dealii::Vector<double> calcRes(5);
//    l.Jac.vmult(calcRes, l.dLambda);
//    calcRes -= l.Res;
//    std::cout << calcRes << std::endl;

    unsigned int iter = l.invertQ(new_Q);
    std::cout << "Total iterations were: " << iter << std::endl;
    std::cout << l.Lambda << std::endl;
    std::cout << l.dLambda << std::endl;
    std::cout << l.Res << std::endl;

    dealii::Vector<double> new_Q2({6.0/10.0,0.0,0.0,-3.0/10.0,0.0});
    iter = l.invertQ(new_Q2);
    std::cout << "Total iterations were: " << iter << std::endl;
    std::cout << l.Lambda << std::endl;

    return 0;
}
