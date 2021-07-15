#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/table_indices.h>
#include "sphere_lebedev_rule.hpp"

// Have to declare static variables here to use outside of class definitions
// -- quirk of C++11
template <int order>
constexpr int LagrangeMultiplier<order>::vec_dim;
template <int order>
constexpr int LagrangeMultiplier<order>::mat_dim;
template <int order>
constexpr std::array<int, LagrangeMultiplier<order>::vec_dim>
	LagrangeMultiplier<order>::Q_row;
template <int order>
constexpr std::array<int, LagrangeMultiplier<order>::vec_dim>
	LagrangeMultiplier<order>::Q_col;
template <int order>
constexpr std::array<std::array<int, LagrangeMultiplier<order>::mat_dim>,
    				 LagrangeMultiplier<order>::mat_dim>
	LagrangeMultiplier<order>::Q_idx;

template <int order>
const std::vector<dealii::Point<LagrangeMultiplier<order>::mat_dim>>
	LagrangeMultiplier<order>::lebedev_coords = makeLebedevCoords();
template <int order>
const std::vector<double>
	LagrangeMultiplier<order>::lebedev_weights = makeLebedevWeights();



template <int order>
std::vector<dealii::Point<LagrangeMultiplier<order>::mat_dim>>
LagrangeMultiplier<order>::makeLebedevCoords()
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    std::vector<dealii::Point<mat_dim>> coords;
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



template <int order>
std::vector<double>
LagrangeMultiplier<order>::makeLebedevWeights()
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



template <int order>
LagrangeMultiplier<order>::LagrangeMultiplier(double in_alpha,
        									  double in_tol,
											  unsigned int in_max_iter)
	: alpha(in_alpha)
	, tol(in_tol)
	, max_iter(in_max_iter)
	, Jac(LagrangeMultiplier<order>::vec_dim,
		  LagrangeMultiplier<order>::vec_dim)
{
    assert(alpha <= 1);
    Lambda.reinit(vec_dim);
    Q.reinit(vec_dim);
    Res.reinit(vec_dim);
}



template <int order>
void LagrangeMultiplier<order>::setQ(dealii::Vector<double> &new_Q)
{
    Q = new_Q;
}



template <int order>
double LagrangeMultiplier<order>::sphereIntegral(
        std::function<double
        (dealii::Point<LagrangeMultiplier<order>::mat_dim>)> integrand)
{
    double integral;
    for (int k=0; k<order; ++k) {
        integral += lebedev_weights[k]*integrand(lebedev_coords[k]);
    }
    integral *= 4*M_PI;

    return integral;
}



template <int order>
double LagrangeMultiplier<order>::lambdaSum(
        dealii::Point<LagrangeMultiplier<order>::mat_dim> x)
{
    // Fill in lower triangle
    double sum = 0;
    for (int k = 0; k < mat_dim; ++k) {
        for (int l = 0; l < k; ++l) {
            sum += Lambda[Q_idx[k][l]] * x[k]*x[l];
        }
    }
    // Multiply by 2 to get upper triangle contribution
    sum *= 2;

    // Get diagonal contributions
    sum += Lambda[Q_idx[0][0]] * x[0]*x[0];
    sum += Lambda[Q_idx[1][1]] * x[1]*x[1];
    sum -= (Lambda[Q_idx[0][0]] + Lambda[Q_idx[1][1]]) * x[2]*x[2];

    return sum;
}



template <int order>
void LagrangeMultiplier<order>::updateRes()
{
    auto denomIntegrand =
        [this](dealii::Point<mat_dim> x)
        {return exp(lambdaSum(x));};
    double Z = sphereIntegral(denomIntegrand);

    for (int m = 0; m < vec_dim; ++m) {
        auto numIntegrand =
            [this, m](dealii::Point<mat_dim> x)
            {return x[Q_row[m]]*x[Q_col[m]] * exp(lambdaSum(x));};

        Res[m] = sphereIntegral(numIntegrand) / Z;
        Res[m] -= (1.0/3.0)*delta[Q_row[m]][Q_col[m]];
        Res[m] -= Q[m];
    }
}



template <int order>
void LagrangeMultiplier<order>::updateJac()
{
    Jac.reinit(vec_dim, vec_dim);
    
    auto denomIntegrand =
        [this](dealii::Point<mat_dim> x)
        {return exp(lambdaSum(x));};
    double Z = sphereIntegral(denomIntegrand);

    // Calculate each entry in Jacobian by calculating
    // relevant integrals around the sphere
    for (int m = 0; m < vec_dim; ++m) {
        for (int n = 0; n < vec_dim; ++n) {
            auto numIntegrand1 =
                [this, m](dealii::Point<mat_dim> x)
                {return x[Q_row[m]]*x[Q_col[m]]*exp(lambdaSum(x));};
            double int1 = sphereIntegral(numIntegrand1);

            // Need to treat diagonal elements and off-diagonal
            // elements differently
            double int2;
            double int3;
            if (n == 0 || n == 3) {
                auto numIntegrand2 =
                    [this, m, n](dealii::Point<mat_dim> x)
                    {return x[Q_row[m]]*x[Q_col[m]]
                        *(x[Q_row[n]]*x[Q_row[n]] - x[2]*x[2])
                            *exp(lambdaSum(x));};
                auto numIntegrand3 =
                    [this, n](dealii::Point<mat_dim> x)
                    {return (x[Q_row[n]]*x[Q_row[n]] - x[2]*x[2])
                        *exp(lambdaSum(x));};

                int2 = sphereIntegral(numIntegrand2);
                int3 = sphereIntegral(numIntegrand3);
            } else {
                auto numIntegrand2 =
                    [this, m, n](dealii::Point<mat_dim> x)
                    {return x[Q_row[m]]*x[Q_col[m]]
                        *x[Q_row[n]]*x[Q_col[n]]
                            *exp(lambdaSum(x));};
                auto numIntegrand3 =
                    [this, n](dealii::Point<mat_dim> x)
                    {return x[Q_row[n]]*x[Q_col[n]] * exp(lambdaSum(x));};

                int2 = 2*sphereIntegral(numIntegrand2);
                int3 = 2*sphereIntegral(numIntegrand3);
            }

            Jac(m, n) = (int2 / Z) - (int1*int3 / (Z*Z));
        }
    }
}



template <int order>
void LagrangeMultiplier<order>::updateVariation()
{
    Jac.compute_lu_factorization();
    dLambda = Res;
    Jac.solve(dLambda);
}



template <int order>
unsigned int LagrangeMultiplier<order>::invertQ(dealii::Vector<double> &new_Q)
{
    this->setQ(new_Q);
    this->updateRes();

    unsigned int iter = 0;
    while (Res.l2_norm() > tol && iter < max_iter) {
        this->updateJac();
        std::cout << std::endl;
        this->updateVariation();
        dLambda *= alpha;
        Lambda -= dLambda;
        this->updateRes();
        std::cout << Res.l2_norm() << std::endl;

        ++iter;
    }

    return iter;
}



template <int order>
void LagrangeMultiplier<order>::printVecTest(
        std::function<double
					  (dealii::Point<LagrangeMultiplier<order>::mat_dim>)> f
		)
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];
    ld_by_order(order, x, y, z, w);
  
    int sum = 0;
    for (int k = 0; k < order; ++k) {
        sum += abs(lebedev_coords[k][0] - x[k]);
        sum += abs(lebedev_coords[k][1] - y[k]);
        sum += abs(lebedev_coords[k][2] - z[k]);
        sum += abs(lebedev_weights[k] - w[k]);
    }

    std::cout << "Sum is: " << sum << std::endl; 
    delete x;
    delete y;
    delete z;
    delete w;

    double integral = sphereIntegral(f);
    std::cout << "Integral is: " << integral << std::endl;
    std::cout << "Integral is supposed to be: "
        << M_PI*M_PI << std::endl;
}

#include "LagrangeMultiplier.inst"
