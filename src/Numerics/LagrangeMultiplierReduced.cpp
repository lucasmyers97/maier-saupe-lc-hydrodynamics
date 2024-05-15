#include "LagrangeMultiplierReduced.hpp"

#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <tuple>

#include "sphere_lebedev_rule/sphere_lebedev_rule.hpp"

LagrangeMultiplierReduced::
LagrangeMultiplierReduced(const int order_,
                          const double alpha_,
                          const double tol_,
                          const unsigned int max_iter_)
    : inverted(false)
    , Jac_updated(false)
    , alpha(alpha_)
    , tol(tol_)
    , max_iter(max_iter_)
    , leb(makeLebedevCoords(order_))
{
    if (alpha > 1.0)
        throw std::invalid_argument("alpha > 1 in LagrangeMultiplierReduced");
}



dealii::Tensor<1, 2, double> LagrangeMultiplierReduced::returnLambda() const
{
    assert(inverted);
    return Lambda;
}



dealii::Tensor<2, 2, double> LagrangeMultiplierReduced::returnJac()
{
    assert(inverted);
    if (!Jac_updated)
        updateResJac();

    return Jac;
}



double LagrangeMultiplierReduced::returnZ() const
{
    assert(inverted);

    /* In the actual calculation we neglect this factor since it shows up in
     * Z and the expression for Q (so it cancels out).
     * We need to include it here because it will change the actual energy.
     * The 4 pi is actually not needed (since it's just a constant because Z
     * comes as part of a log, but we include it here anyways for consistency).
     */
    return Z * std::exp(-(Lambda[0] + Lambda[1])) * 4 * dealii::numbers::PI;
}




unsigned int LagrangeMultiplierReduced::
invertQ(const dealii::Tensor<1, 2, double> &Q_in)
{
    initializeInversion(Q_in);

    unsigned int iter = 0;
    while (Res.norm() > tol && iter < max_iter)
    {
        this->updateVariation();
        Lambda += alpha * dLambda;
        this->updateResJac();

        ++iter;
    }
    inverted = (Res.norm() < tol);

    if (!inverted)
        throw std::runtime_error("Could not invert LagrangeMultiplierReduced");

    return iter;
}



void LagrangeMultiplierReduced::setOrder(const int order)
{
    leb = makeLebedevCoords(order);
}



void LagrangeMultiplierReduced::
initializeInversion(const dealii::Tensor<1, 2, double> &Q_in)
{
    inverted = false;

    Q = Q_in;
    Lambda = 0;
    Res = 0;
    Res -= Q; // can explicitly compute for Lambda = 0

    // can also explicitly compute for Lambda = 0
    Jac = 0;
    Jac[0][0] = 2.0 / 15.0;
    Jac[1][1] = 2.0 / 15.0;
}



void LagrangeMultiplierReduced::updateResJac()
{
    double x_x = 0;
    double y_y = 0;
    double w = 0;
    double A = 2 * Lambda[0] + Lambda[1];
    double B = Lambda[0] + 2 * Lambda[1];

    double exp_lambda_w = 0;
    double x_int = 0;
    double y_int = 0;
    double xx_int = 0;
    double yy_int = 0;
    double xy_int = 0;

    Z = 0;
    Res = 0;
    Jac = 0;

    // Calculate each term in Lebedev quadrature for each integral, add to total
    // quadrature value until we've summed all terms
    #pragma unroll
    for (int quad_idx = 0; quad_idx < leb.x.size(); ++quad_idx)
    {
        x_x = leb.x[quad_idx];
        x_x *= x_x;
        y_y = leb.y[quad_idx];
        y_y *= y_y;

        /* this is off by a factor of 4 \pi (because of the Lebedev quadrature)
         * and also a factor of e^((A + B)/3).
         * We correct for these upon return of Z
         */
        exp_lambda_w = std::exp( A*x_x + B*y_y ) * leb.w[quad_idx];

        Z += exp_lambda_w;
        x_int += x_x * exp_lambda_w;
        y_int += y_y * exp_lambda_w;
        xx_int += x_x*x_x * exp_lambda_w;
        yy_int += y_y*y_y * exp_lambda_w;
        xy_int += x_x*y_y * exp_lambda_w;
		}

    double Z_inv = (1 / Z);

    Res[0] = Z_inv * x_int - (1.0 / 3.0) - Q[0];
    Res[1] = Z_inv * y_int - (1.0 / 3.0) - Q[1];

    Jac[0][0] = Z_inv * (2*xx_int + xy_int - x_int)
                - Z_inv*Z_inv * (x_int * (2*x_int + y_int - Z));
    Jac[1][0] = Z_inv * (2 * xy_int + yy_int - y_int)
                - Z_inv*Z_inv * (y_int * (2 * x_int + y_int - Z));
    Jac[0][1] = Z_inv * (2 * xy_int + xx_int - x_int)
                - Z_inv*Z_inv * (x_int * (2 * y_int + x_int - Z));
    Jac[1][1] = Z_inv * (2 * yy_int + xy_int - y_int)
                - Z_inv*Z_inv * (y_int * (2 * y_int + x_int - Z));

    Jac_updated = true;
}



void LagrangeMultiplierReduced::updateVariation()
{
    dealii::Tensor<2, 2, double> Jac_inverse;
    Jac_inverse = dealii::invert(Jac);
    Jac_updated = false; // Can't use Jac when it's inverted
    dLambda = -(Jac_inverse * Res);
}



LagrangeMultiplierReduced::ReducedLebedevCoords
LagrangeMultiplierReduced::makeLebedevCoords(int order)
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    ReducedLebedevCoords leb_;

    unsigned int x_zero = 0;
    unsigned int y_zero = 0;
    unsigned int z_zero = 0;
    unsigned int num_zeros = 0;

    for (int k = 0; k < order; ++k)
    {
        if ((x[k] < 0) || (y[k] < 0) || (z[k] < 0))
            continue;

        leb_.x.emplace_back(x[k]);
        leb_.y.emplace_back(y[k]);

        // determine weight depending on symmetry
        if (x[k] == 0)
            x_zero = 1;
        if (y[k] == 0)
            y_zero = 1;
        if (z[k] == 0)
            z_zero = 1;
        num_zeros = x_zero + y_zero + z_zero;

        switch (num_zeros)
        {
        case 2:
          leb_.w.emplace_back(w[k] * 2);
          break;
        case 1:
          leb_.w.emplace_back(w[k] * 4);
          break;
        case 0:
          leb_.w.emplace_back(w[k] * 8);
          break;
        }

        x_zero = 0;
        y_zero = 0;
        z_zero = 0;
    }

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;

    return leb_;
}
