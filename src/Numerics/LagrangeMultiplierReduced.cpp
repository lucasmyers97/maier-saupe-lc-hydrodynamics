#include "LagrangeMultiplierReduced.hpp"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "Utilities/maier_saupe_constants.hpp"
#include "sphere_lebedev_rule/sphere_lebedev_rule.hpp"

namespace msc = maier_saupe_constants;

// Utility functions for initializing Lebedev quadrature points & weights
template <int order, int space_dim>
const std::vector<dealii::Point<msc::mat_dim<space_dim>>>
	LagrangeMultiplierReduced<order, space_dim>::
    lebedev_coords = makeLebedevCoords();

template <int order, int space_dim>
const std::vector<double>
    LagrangeMultiplierReduced<order, space_dim>::
    lebedev_weights = makeLebedevWeights();



template <int order, int space_dim>
LagrangeMultiplierReduced<order, space_dim>::
LagrangeMultiplierReduced(const double alpha_,
                          const double tol_,
                          const unsigned int max_iter_)
    : inverted(false)
    , Jac_updated(false)
    , alpha(alpha_)
    , tol(tol_)
    , max_iter(max_iter_)
{
    if (alpha > 1.0)
        throw std::invalid_argument("alpha > 1 in LagrangeMultiplierReduced");
}



template <int order, int space_dim>
dealii::Tensor<1, 2, double> LagrangeMultiplierReduced<order, space_dim>::
returnLambda() const
{
    assert(inverted);
    return Lambda;
}



template <int order, int space_dim>
dealii::Tensor<2, 2, double> LagrangeMultiplierReduced<order, space_dim>::
returnJac()
{
    assert(inverted);
    if (!Jac_updated)
        updateResJac();

    return Jac;
}



template <int order, int space_dim>
double LagrangeMultiplierReduced<order, space_dim>::
returnZ() const
{
    assert(inverted);

    return Z;
}




template <int order, int space_dim>
unsigned int LagrangeMultiplierReduced<order, space_dim>::
invertQ(const dealii::Tensor<1, 2, double> &Q_in)
{
    // TODO: add flag to reinitialize LagrangeMultiplierReduced or not
    // TODO: figure out how to reuse Jacobian easily
    initializeInversion(Q_in);

    // Run Newton's method until residual < tolerance or reach max iterations
    unsigned int iter = 0;
    while (Res.norm() > tol && iter < max_iter)
    {
        this->updateVariation();
        Lambda += alpha * dLambda;
        this->updateResJac();

        ++iter;
    }
    inverted = (Res.norm() < tol);
    assert(inverted);

    return iter;
}



template<int order, int space_dim>
void LagrangeMultiplierReduced<order, space_dim>::
initializeInversion(const dealii::Tensor<1, 2, double> &Q_in)
{
    inverted = false;

    Q = Q_in;
    Lambda = 0;
    Res = 0;
    Res -= Q; // can explicitly compute for Lambda = 0

    // for Jacobian, compute 2/15 on diagonal, 0 elsewhere for Lambda = 0
    Jac = 0;
    Jac[0][0] = 2.0 / 15.0;
    Jac[1][1] = 2.0 / 15.0;
}



template<int order, int space_dim>
void LagrangeMultiplierReduced<order, space_dim>::
updateResJac()
{
    double x = 0;
    double y = 0;
    double w = 0;
    double A = 2 * Lambda[0] + Lambda[1];
    double B = Lambda[0] + 2 * Lambda[1];

    double exp_lambda = 0;
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
    for (int quad_idx = 0; quad_idx < order; ++quad_idx)
    {
        x = lebedev_coords[quad_idx][0];
        y = lebedev_coords[quad_idx][1];
        w = lebedev_weights[quad_idx];

        exp_lambda = std::exp( A*x*x + B*y*y );

        Z += exp_lambda * w;
        x_int += x*x * exp_lambda * w;
        y_int += y*y * exp_lambda * w;
        xx_int += x*x*x*x * exp_lambda * w;
        yy_int += y*y*y*y * exp_lambda *w;
        xy_int += x*y*x*y * exp_lambda * w;
		}

    Res[0] = (1 / Z) * x_int - (1.0 / 3.0) - Q[0];
    Res[1] = (1 / Z) * y_int - (1.0 / 3.0) - Q[1];

    Jac[0][0] = (1 / Z) * (2*xx_int + xy_int - x_int)
                - (1 / (Z*Z)) * (x_int * (2*x_int + y_int - Z));
    Jac[1][0] = (1 / Z) * (2 * xy_int + yy_int - y_int)
                - (1 / (Z * Z)) * (y_int * (2 * x_int + y_int - Z));
    Jac[0][1] = (1 / Z) * (2 * xy_int + xx_int - x_int)
                - (1 / (Z * Z)) * (x_int * (2 * y_int + x_int - Z));
    Jac[1][1] = (1 / Z) * (2 * yy_int + xy_int - y_int)
                - (1 / (Z * Z)) * (y_int * (2 * y_int + x_int - Z));

    Jac_updated = true;
}



template <int order, int space_dim>
void LagrangeMultiplierReduced<order, space_dim>::
updateVariation()
{
    dealii::Tensor<2, 2, double> Jac_inverse;
    Jac_inverse = dealii::invert(Jac);
    Jac_updated = false; // Can't use Jac when it's inverted
    dLambda = -(Jac_inverse * Res);
}



template <int order, int space_dim>
std::vector< dealii::Point<msc::mat_dim<space_dim>> >
LagrangeMultiplierReduced<order, space_dim>::
makeLebedevCoords()
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    std::vector< dealii::Point<msc::mat_dim<space_dim>> > coords;
    coords.reserve(order);
    for (int k = 0; k < order; ++k) {
        coords[k][0] = x[k];
        coords[k][1] = y[k];
        coords[k][2] = z[k];
    }

    delete x;
    delete y;
    delete z;
    delete w;

    return coords;
}



template <int order, int space_dim>
std::vector<double> LagrangeMultiplierReduced<order, space_dim>::
makeLebedevWeights()
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    std::vector<double> weights;
    weights.reserve(order);
    for (int k = 0; k < order; ++k) {
        weights[k] = w[k];
    }

    delete x;
    delete y;
    delete z;
    delete w;

    return weights;
}

#include "LagrangeMultiplierReduced.inst"
