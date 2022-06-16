#include "LagrangeMultiplier.hpp"

#include <iostream>
#include <cmath>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include "Utilities/maier_saupe_constants.hpp"
#include "sphere_lebedev_rule/sphere_lebedev_rule.hpp"

namespace msc = maier_saupe_constants;


template <int dim>
LagrangeMultiplier<dim>::
LagrangeMultiplier(const int order_,
                   const double alpha_,
                   const double tol_,
                   const unsigned int max_iter_)
    : leb(makeLebedevCoords(order_))
    , inverted(false)
    , Jac_updated(false)
    , alpha(alpha_)
    , tol(tol_)
    , max_iter(max_iter_)
    , Jac(msc::vec_dim<dim>,
          msc::vec_dim<dim>)
    , Z(0)
{
    assert(alpha <= 1);
    Q.reinit(msc::vec_dim<dim>);
    Lambda.reinit(msc::vec_dim<dim>);
    Res.reinit(msc::vec_dim<dim>);
}



template <int dim>
void LagrangeMultiplier<dim>::
returnLambda(dealii::Vector<double> &outLambda) const
{
    Assert(inverted, dealii::ExcInternalError());
    outLambda = Lambda;
}


template <int dim>
void LagrangeMultiplier<dim>::
returnJac(dealii::LAPACKFullMatrix<double> &outJac)
{
    Assert(inverted, dealii::ExcInternalError());
    if (!Jac_updated) { updateResJac(); }
    outJac = Jac;
}



template <int dim>
double LagrangeMultiplier<dim>::
returnZ() const
{
    Assert(inverted, dealii::ExcInternalError());
    return Z;
}




template <int dim>
unsigned int LagrangeMultiplier<dim>::
invertQ(const dealii::Vector<double> &Q_in)
{
    // TODO: add flag to reinitialize LagrangeMultiplier or not
    // TODO: figure out how to reuse Jacobian easily
    initializeInversion(Q_in);

    // Run Newton's method until residual < tolerance or reach max iterations
    unsigned int iter = 0;
    while (Res.l2_norm() > tol && iter < max_iter)
    {
        this->updateVariation();
        dLambda *= alpha;
        Lambda -= dLambda;
        this->updateResJac();

        ++iter;
    }
    inverted = (Res.l2_norm() < tol);
    Assert(inverted, dealii::ExcInternalError())

    return iter;
}



template<int dim>
void LagrangeMultiplier<dim>::
initializeInversion(const dealii::Vector<double> &Q_in)
{
    inverted = false;

    Q = Q_in;
    Lambda = 0;
    Res = 0;
    Res -= Q; // can explicitly compute for Lambda = 0

    // for Jacobian, compute 2/15 on diagonal, 0 elsewhere for Lambda = 0
    for (int i = 0; i < msc::vec_dim<dim>; ++i)
        for (int j = 0; j < msc::vec_dim<dim>; ++j)
        {
            if (i == j)
                Jac(i, j) = 2.0 / 15.0;
            else
                Jac(i, j) = 0;
        }
}



template<int dim>
void LagrangeMultiplier<dim>::
updateResJac()
{
	double exp_lambda{0};
    Z = 0;
    Res = 0;
    Jac = 0;

    int1.fill(0);
    for (auto &row : int2)
        row.fill(0);
    for (auto &row : int3)
        row.fill(0);
    int4.fill(0);

	// Calculate each term in Lebedev quadrature for each integral, add to total
	// quadrature value until we've summed all terms
    for (int quad_idx = 0; quad_idx < leb.w.size(); ++quad_idx)
    {
        exp_lambda = std::exp( lambdaSum(leb.x[quad_idx]) );

        Z += exp_lambda * leb.w[quad_idx];

        #pragma unroll
        for (int m = 0; m < msc::vec_dim<dim>; ++m)
        {
            int1[m] += calcInt1Term(exp_lambda, quad_idx,
                                    msc::Q_row<dim>[m],
                                    msc::Q_col<dim>[m]);
            int4[m] += calcInt4Term(exp_lambda, quad_idx,
                                    msc::Q_row<dim>[m],
                                    msc::Q_col<dim>[m]);

			      #pragma unroll
            for (int n = 0; n < msc::vec_dim<dim>; ++n)
            {
                int2[m][n] += calcInt2Term(exp_lambda, quad_idx,
                                           msc::Q_row<dim>[m],
                                           msc::Q_col<dim>[m],
                                           msc::Q_row<dim>[n],
                                           msc::Q_col<dim>[n]);
                int3[m][n] += calcInt3Term(exp_lambda, quad_idx,
                                           msc::Q_row<dim>[m],
                                           msc::Q_col<dim>[m],
                                           msc::Q_row<dim>[n],
                                           msc::Q_col<dim>[n]);
            }
        }
    }

    // Calculate each entry of residual and Jacobian using integral values
    #pragma unroll
    for (int m = 0; m < msc::vec_dim<dim>; ++m)
    {
        Res[m] = int1[m] / Z
            - (1.0 / 3.0) * msc::delta_vec<dim>[m]
            - Q[m];

        #pragma unroll
        for (int n = 0; n < msc::vec_dim<dim>; ++n)
        {
            if (n == 0 || n == 3)
                Jac(m, n) = int3[m][n] / Z
                    - int1[m] * int4[n] / (Z*Z);
            else
                Jac(m, n) = 2 * int2[m][n] / Z
                    - 2 * int1[m] * int1[n] / (Z*Z);
        }
    }
    Jac_updated = true;
}



template <int dim>
double LagrangeMultiplier<dim>::calcInt1Term
(const double exp_lambda, const int quad_idx,
 const int i_m, const int j_m) const
{
    return exp_lambda * leb.w[quad_idx]
        * leb.x[quad_idx][i_m]
        * leb.x[quad_idx][j_m];
}



template <int dim>
double LagrangeMultiplier<dim>::calcInt2Term
(const double exp_lambda, const int quad_idx,
 const int i_m, const int j_m, const int i_n, const int j_n) const
{
    return exp_lambda * leb.w[quad_idx]
        * leb.x[quad_idx][i_m]
        * leb.x[quad_idx][j_m]
        * leb.x[quad_idx][i_n]
        * leb.x[quad_idx][j_n];
}



template <int dim>
double LagrangeMultiplier<dim>::calcInt3Term
(const double exp_lambda, const int quad_idx,
 const int i_m, const int j_m, const int i_n, const int j_n) const
{
    return exp_lambda * leb.w[quad_idx]
        * leb.x[quad_idx][i_m]
        * leb.x[quad_idx][j_m]
        * (leb.x[quad_idx][i_n]
           * leb.x[quad_idx][i_n]
           -
           leb.x[quad_idx][2]
           * leb.x[quad_idx][2]);
}



template <int dim>
double LagrangeMultiplier<dim>::calcInt4Term
(const double exp_lambda, const int quad_idx,
 const int i_m, const int j_m) const
{
    return exp_lambda * leb.w[quad_idx]
        * (leb.x[quad_idx][i_m]
           * leb.x[quad_idx][i_m]
           -
           leb.x[quad_idx][2]
           * leb.x[quad_idx][2]);
}



template <int dim>
void LagrangeMultiplier<dim>::
updateVariation()
{
    Jac.compute_lu_factorization();
    Jac_updated = false; // Can't use Jac when it's lu-factorized
    dLambda = Res; // LAPACK syntax: put rhs into vec which will hold solution
    Jac.solve(dLambda); // LAPACK puts solution back into input vector
}



template <int dim>
double LagrangeMultiplier<dim>::
lambdaSum(const dealii::Point<msc::mat_dim<dim>> x) const
{
    // Calculates \xi_i \Lambda_{ij} \xi_j

    // Sum lower triangle
    double sum = 0;
    for (int k = 0; k < msc::mat_dim<dim>; ++k)
        for (int l = 0; l < k; ++l)
            sum += Lambda[msc::Q_idx<dim>[k][l]] * x[k]*x[l];

    // Multiply by 2 to get upper triangle contribution
    sum *= 2;

    // Get diagonal contributions
    sum += Lambda[msc::Q_idx<dim>[0][0]] * x[0]*x[0];
    sum += Lambda[msc::Q_idx<dim>[1][1]] * x[1]*x[1];
    sum -= ( Lambda[msc::Q_idx<dim>[0][0]]
             + Lambda[msc::Q_idx<dim>[1][1]] ) * x[2]*x[2];

    return sum;
}



template <int dim>
typename LagrangeMultiplier<dim>::LebedevCoords
LagrangeMultiplier<dim>::makeLebedevCoords(const int order_)
{
    double *x, *y, *z, *w;
    x = new double[order_];
    y = new double[order_];
    z = new double[order_];
    w = new double[order_];

    ld_by_order(order_, x, y, z, w);

    LebedevCoords leb_;
    leb_.x.resize(order_);
    leb_.w.resize(order_);
    for (int k = 0; k < order_; ++k)
    {
        leb_.x[k][0] = x[k];
        leb_.x[k][1] = y[k];
        leb_.x[k][2] = z[k];
        leb_.w[k] = w[k];
    }

    delete x;
    delete y;
    delete z;
    delete w;

    return leb_;
}

template class LagrangeMultiplier<2>;
template class LagrangeMultiplier<3>;
