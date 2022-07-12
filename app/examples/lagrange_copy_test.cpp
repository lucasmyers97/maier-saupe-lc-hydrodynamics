/**
 * This program tries to copy a LagrangeMultiplierAnalytic object to see whether
 * a default copy constructor and assignment operator are created by the
 * compiler.
 */

#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include <deal.II/lac/vector.h>

int main()
{
    constexpr int dim = 2;
    const int order = 974;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const unsigned int max_iter = 20;

    LagrangeMultiplierAnalytic<dim> lma(order, alpha, tol, max_iter);
    LagrangeMultiplierAnalytic<dim> lma2 = lma;

    dealii::Vector<double> Q_in(5);
    Q_in[0] = 0.1;
    Q_in[1] = 0.05;
    Q_in[3] = -0.05;

    lma.invertQ(Q_in);

    return 0;
}
