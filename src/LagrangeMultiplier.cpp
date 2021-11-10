#include "LagrangeMultiplier.hpp"
#include "maier_saupe_constants.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/table_indices.h>
#include "maier_saupe_constants.hpp"
#include "sphere_lebedev_rule.hpp"

using namespace maier_saupe_constants;

// Utility functions for initializing Lebedev quadrature points & weights
template <int order, int space_dim>
const std::vector<dealii::Point<mat_dim<space_dim>>>
	LagrangeMultiplier<order, space_dim>::
    lebedev_coords = makeLebedevCoords();

template <int order, int space_dim>
const std::vector<double>
    LagrangeMultiplier<order, space_dim>::
    lebedev_weights = makeLebedevWeights();



template <int order, int space_dim>
LagrangeMultiplier<order, space_dim>::
LagrangeMultiplier(double alpha_, double tol_, unsigned int max_iter_)
	: inverted(false)
	, Jac_updated(false)
	, alpha(alpha_)
	, tol(tol_)
	, max_iter(max_iter_)
	, Jac(vec_dim<space_dim>,
		  vec_dim<space_dim>)
{
    assert(alpha <= 1);
    Lambda.reinit(vec_dim<space_dim>);
    Q.reinit(vec_dim<space_dim>);
    Res.reinit(vec_dim<space_dim>);
}



template <int order, int space_dim>
void LagrangeMultiplier<order, space_dim>::
returnLambda(dealii::Vector<double> &outLambda)
{
	Assert(inverted, dealii::ExcInternalError());
	outLambda = Lambda;
}


template <int order, int space_dim>
void LagrangeMultiplier<order, space_dim>::
returnJac(dealii::LAPACKFullMatrix<double> &outJac)
{
	Assert(inverted, dealii::ExcInternalError());
	if (!Jac_updated) { updateResJac(); }
	outJac = Jac;
}



template <int order, int space_dim>
unsigned int LagrangeMultiplier<order, space_dim>::
invertQ(dealii::Vector<double> &Q_in)
{
    initializeInversion(Q_in);

    // Run Newton's method until residual < tolerance or reach max iterations
    unsigned int iter = 0;
    while (Res.l2_norm() > tol && iter < max_iter) {
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



template<int order, int space_dim>
void LagrangeMultiplier<order, space_dim>::
initializeInversion(dealii::Vector<double> &Q_in)
{
    inverted = false;

    Q = Q_in;
    Lambda = 0;
    Res = 0;
    Res -= Q; // can explicitly compute for Lambda = 0
	
	// for Jacobian, compute 2/15 on diagonal, 0 elsewhere for Lambda = 0
	for (int i = 0; i < vec_dim<space_dim>; ++i)
		for (int j = 0; j < vec_dim<space_dim>; ++j)
		{
			if (i == j)
				Jac(i, j) = 2.0 / 15.0;
			else
				Jac(i, j) = 0;
		}
}



template<int order, int space_dim>
void LagrangeMultiplier<order, space_dim>::
updateResJac()
{
	double exp_lambda{};
    Z = 0;
    Res = 0;
    Jac = 0;

    int1 = {0};
    int2 = {0};
    int3 = {0};
    int4 = {0};
	
	// Calculate each term in Lebedev quadrature for each integral, add to total
	// quadrature value until we've summed all terms
	#pragma unroll
	for (int quad_idx = 0; quad_idx < order; ++quad_idx)
	{
		exp_lambda = std::exp( lambdaSum(lebedev_coords[quad_idx]) );
		
		Z += exp_lambda * lebedev_weights[quad_idx];
		
		#pragma unroll
		for (int m = 0; m < vec_dim<space_dim>; ++m)
		{
			int1[m] += calcInt1Term(exp_lambda, quad_idx, 
                                    Q_row<space_dim>[m], Q_col<space_dim>[m]);
			int4[m] += calcInt4Term(exp_lambda, quad_idx, 
                                    Q_row<space_dim>[m], Q_col<space_dim>[m]);
			
			#pragma unroll
			for (int n = 0; n < vec_dim<space_dim>; ++n)
			{
				int2[m][n] += calcInt2Term
                                (exp_lambda, quad_idx, 
								 Q_row<space_dim>[m], Q_col<space_dim>[m], 
								 Q_row<space_dim>[n], Q_col<space_dim>[n]);
				int3[m][n] += calcInt3Term
                                (exp_lambda, quad_idx, 
								 Q_row<space_dim>[m], Q_col<space_dim>[m], 
								 Q_row<space_dim>[n], Q_col<space_dim>[n]);
			}
		}
	}
	
	// Calculate each entry of residual and Jacobian using integral values
	#pragma unroll
	for (int m = 0; m < vec_dim<space_dim>; ++m)
	{
		Res[m] = int1[m] / Z 
				 - (1.0 / 3.0) * delta_vec<space_dim>[m]
				 - Q[m];
		
		#pragma unroll
		for (int n = 0; n < vec_dim<space_dim>; ++n)
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



template <int order, int space_dim>
double LagrangeMultiplier<order, space_dim>::
calcInt1Term
(const double exp_lambda, const int quad_idx, const int i_m, const int j_m)
{
	return exp_lambda * lebedev_weights[quad_idx]
		   * lebedev_coords[quad_idx][i_m]
		   * lebedev_coords[quad_idx][j_m];
}



template <int order, int space_dim>
double LagrangeMultiplier<order, space_dim>::
calcInt2Term
(const double exp_lambda, const int quad_idx,
const int i_m, const int j_m, const int i_n, const int j_n)
{
	return exp_lambda * lebedev_weights[quad_idx]
		   * lebedev_coords[quad_idx][i_m]
		   * lebedev_coords[quad_idx][j_m]
		   * lebedev_coords[quad_idx][i_n]
		   * lebedev_coords[quad_idx][j_n];
}



template <int order, int space_dim>
double LagrangeMultiplier<order, space_dim>::
calcInt3Term
(const double exp_lambda, const int quad_idx,
const int i_m, const int j_m, const int i_n, const int j_n)
{
	return exp_lambda * lebedev_weights[quad_idx]
		   * lebedev_coords[quad_idx][i_m]
		   * lebedev_coords[quad_idx][j_m]
		   * (lebedev_coords[quad_idx][i_n]
			  * lebedev_coords[quad_idx][i_n]
			  - 
			  lebedev_coords[quad_idx][2]
			  * lebedev_coords[quad_idx][2]);
}



template <int order, int space_dim>
double LagrangeMultiplier<order, space_dim>::calcInt4Term
(const double exp_lambda, const int quad_idx, const int i_m, const int j_m)
{
	return exp_lambda * lebedev_weights[quad_idx]
		   * (lebedev_coords[quad_idx][i_m]
			  * lebedev_coords[quad_idx][i_m]
			  - 
			  lebedev_coords[quad_idx][2]
			  * lebedev_coords[quad_idx][2]);
}



template <int order, int space_dim>
void LagrangeMultiplier<order, space_dim>::
updateVariation()
{
    Jac.compute_lu_factorization();
    Jac_updated = false; // Can't use Jac when it's lu-factorized
    dLambda = Res; // LAPACK syntax: put rhs into vec which will hold solution
    Jac.solve(dLambda); // LAPACK puts solution back into input vector
}



template <int order, int space_dim>
double LagrangeMultiplier<order, space_dim>::
lambdaSum(dealii::Point<mat_dim<space_dim>> x)
{
	// Calculates \xi_i \Lambda_{ij} \xi_j

    // Sum lower triangle
    double sum = 0;
    for (int k = 0; k < mat_dim<space_dim>; ++k) {
        for (int l = 0; l < k; ++l) {
            sum += Lambda[Q_idx<space_dim>[k][l]] * x[k]*x[l];
        }
    }
    // Multiply by 2 to get upper triangle contribution
    sum *= 2;

    // Get diagonal contributions
    sum += Lambda[Q_idx<space_dim>[0][0]] * x[0]*x[0];
    sum += Lambda[Q_idx<space_dim>[1][1]] * x[1]*x[1];
    sum -= ( Lambda[Q_idx<space_dim>[0][0]] 
             + Lambda[Q_idx<space_dim>[1][1]] ) * x[2]*x[2];

    return sum;
}



// template <int order, int space_dim>
// void LagrangeMultiplier<order, space_dim>::
// lagrangeTest()
// {
// 	auto f = [](dealii::Point<mat_dim<space_dim>> x)
// 			   { return sqrt(x[0]*x[0] + x[1]*x[1]); };
// 	printVecTest(f);

// 	// Calc 5*5*exp(0)
//     dealii::Point<3> x{1.0, 5.0, 5.0};
//     int m = 2;
//     double y = x[m];
//     auto numIntegrand =
//         [this, y](dealii::Point<3> p)
//         {return y*y * exp(lambdaSum(p));};
//     std::cout << "5*5 = " << numIntegrand(x) << std::endl;

//     // Try inversion with Q in physical bounds
//     dealii::Vector<double> new_Q({2.0/4.0 - 1e-2,0.0,0.0,-1.0/4.0,0.0});

//     setQ(new_Q);
//     updateRes();
//     updateJac();

//     std::cout << "R = " << Res << std::endl;
//     std::cout << "Jac = ";
//     Jac.print_formatted(std::cout);
//     std::cout << std::endl;

//     updateVariation();
//     std::cout << "dLambda = " << dLambda << std::endl;
//     std::cout << std::endl;

//     updateJac();

//     // Invert Q altogether
//     unsigned int iter = invertQ();
//     std::cout << "Total number of iterations was: " << iter << std::endl;
//     std::cout << "Lambda = " << Lambda << std::endl;
//     std::cout << "Residual is: " << Res << std::endl;
//     std::cout << std::endl;

//     // Try with Q close to physical limits
//     dealii::Vector<double> new_Q2({6.0/10.0,0.0,0.0,-3.0/10.0,0.0});
//     setQ(new_Q2);
//     iter = invertQ();
//     std::cout << "Total number of iterations was: " << iter << std::endl;
//     std::cout << "Lambda = " << Lambda << std::endl;
//     std::cout << "Residual is: " << Res << std::endl;
//     std::cout << std::endl;

//     // Try new Q and try to copy
//     setQ(new_Q);
//     dealii::Vector<double> newLambda;
//     returnLambda(newLambda);
//     std::cout << "Lambda = " << newLambda << std::endl;
//     std::cout << std::endl;

//     // Try new Q and copy Jacobian
//     setQ(new_Q);
//     dealii::LAPACKFullMatrix<double> newJac;
//     returnJac(newJac);
//     std::cout << "Jac = " << std::endl;
//     newJac.print_formatted(std::cout);
//     std::cout << std::endl;
// }



// template <int order, int space_dim>
// void LagrangeMultiplier<order, space_dim>::
// printVecTest(std::function<double (dealii::Point<mat_dim<space_dim>>)> f)
// {
//     double *x, *y, *z, *w;
//     x = new double[order];
//     y = new double[order];
//     z = new double[order];
//     w = new double[order];
//     ld_by_order(order, x, y, z, w);
  
//     int sum = 0;
//     for (int k = 0; k < order; ++k) {
//         sum += abs(lebedev_coords[k][0] - x[k]);
//         sum += abs(lebedev_coords[k][1] - y[k]);
//         sum += abs(lebedev_coords[k][2] - z[k]);
//         sum += abs(lebedev_weights[k] - w[k]);
//     }

//     std::cout << "Sum is: " << sum << std::endl; 
//     delete x;
//     delete y;
//     delete z;
//     delete w;

//     double integral = sphereIntegral(f);
//     std::cout << "Integral is: " << integral << std::endl;
//     std::cout << "Integral is supposed to be: "
//         << M_PI*M_PI << std::endl;
// }

template <int order, int space_dim>
std::vector< dealii::Point<mat_dim<space_dim>> >
LagrangeMultiplier<order, space_dim>::
makeLebedevCoords()
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    std::vector<dealii::Point<mat_dim<space_dim>>> coords;
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
std::vector<double> LagrangeMultiplier<order, space_dim>::
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

#include "LagrangeMultiplier.inst"