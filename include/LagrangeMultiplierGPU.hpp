#ifndef LAGRANGE_MULTIPLIER_GPU_HPP
#define LAGRANGE_MULTIPLIER_GPU_HPP

#include <math.h>
#include <assert.h>
#include "LUMatrixGPU.hpp"

namespace{
	constexpr int N_COORDS{3};
	constexpr double big_num{1e15};
}

template <typename T, int order, unsigned int vec_dim>
class LagrangeMultiplierGPU
{
public:
	__device__ LagrangeMultiplierGPU() {};
	__device__ inline void readLebedevGlobal(const T* g_lebedev_coords,
											 const T* g_lebedev_weights,
											 const int t_idx, 
											 const int n_threads,
											 T* s_lebedev_coords,
											 T* s_lebedev_weights);
	__device__ inline void setLebedevData(T* in_lebedev_coords, 
									      T* in_lebedev_weights);
	__device__ inline void setParams(const T tol, const int max_iters);
	__device__ inline void calcLambda(T* Q_in);

private:
	__device__ inline void initializeInversion(const T* Q_in);
	__device__ inline void calcdLambda();
	__device__ inline void calcResJac();
	__device__ inline double calcExpLambda(int row_idx);
	__device__ inline double calcInt1(const double exp_lambda,
									  const int coord_idx, const int row_idx,
			   	   	   	   	   	   	  const int i_m, const int j_m);
	__device__ inline double calcInt2(const double exp_lambda,
									  const int coord_idx, const int row_idx,
			   	   	   	   	   	      const int i_m, const int j_m,
									  const int i_n, const int j_n);
	__device__ inline double calcInt3(const double exp_lambda,
									  const int coord_idx, const int row_idx,
				   	   	   	   	   	  const int i_m, const int j_m,
									  const int i_n, const int j_n);
	__device__ inline double calcInt4(const double exp_lamda,
									  const int coord_idx, const int row_idx,
					   	   	   	   	  const int i_m, const int j_m);


	T* lebedev_coords{NULL};
	T* lebedev_weights{NULL};

	LUMatrixGPU<T, vec_dim> Jac;
	T Res[vec_dim];
	T Q[vec_dim];
	T Lambda[vec_dim] = {0};
	T dLambda[vec_dim];

	T tol;
	int max_iters;
};



template <typename T, int order, unsigned int vec_dim>
__device__ inline
void
LagrangeMultiplierGPU<T, order, vec_dim>::readLebedevGlobal
(const T* g_lebedev_coords, const T* g_lebedev_weights,
 const int t_idx, const int n_threads,
 T* s_lebedev_coords, T* s_lebedev_weights)
{
	#pragma unroll
	for (int i = t_idx; i < order; i += n_threads)
	{
		s_lebedev_weights[i] = g_lebedev_weights[i];
	}

	#pragma unroll
	for (int i = t_idx; i < N_COORDS*order; i += n_threads)
	{
		s_lebedev_coords[i] = g_lebedev_coords[i];
	}
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline void
LagrangeMultiplierGPU<T, order, vec_dim>::setLebedevData
(T* in_lebedev_coords, T* in_lebedev_weights)
{
	lebedev_coords = in_lebedev_coords;
	lebedev_weights = in_lebedev_weights;
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline 
void LagrangeMultiplierGPU<T, order, vec_dim>::setParams
(const T in_tol, const int in_max_iters)
{
	tol = in_tol;
	max_iters = in_max_iters;
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline void 
LagrangeMultiplierGPU<T, order, vec_dim>::calcLambda(T* Q_in)
{
	// Get initial residual, jacobian for given Q
	initializeInversion(Q_in);

	// Calculate norm of initial residual
	double res_norm = 0;
	for (int i = 0; i < vec_dim; ++i)
		res_norm += Res[i]*Res[i];
	res_norm = sqrt(res_norm);

	int iters = 0;
	while (iters < max_iters && res_norm > tol)
	{
		calcdLambda();
		for (int i = 0; i < vec_dim; ++i)
			Lambda[i] -= dLambda[i];

		calcResJac();

		res_norm = 0;
		for (int i = 0; i < vec_dim; ++i)
			res_norm += Res[i]*Res[i];
		res_norm = sqrt(res_norm);

		++iters;
	}
	
	assert(res_norm < tol);
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline void 
LagrangeMultiplierGPU<T, order, vec_dim>::calcdLambda()
{
	Jac.compute_lu_factorization();
	for (int i = 0; i < vec_dim; ++i)
		dLambda[i] = Res[i];
	Jac.solve(dLambda);
}



template<typename T, int order, unsigned int vec_dim>
__device__ inline
void 
LagrangeMultiplierGPU<T, order, vec_dim>::initializeInversion(const T* Q_in)
{
	for (int i = 0; i < vec_dim; ++i)
	{
		Q[i] = Q_in[i];
		Res[i] = -Q[i]; // can explicitly compute for Lambda = 0
	}
	
	// for Jacobian, compute 2/15 on diagonal, 0 elsewhere for Lambda = 0
	for (int i = 0; i < vec_dim; ++i)
		for (int j = 0; j < vec_dim; ++j)
		{
			if (i == j)
				Jac(i, j) = 2.0 / 15.0;
			else
				Jac(i, j) = 0;
		}
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
void
LagrangeMultiplierGPU<T, order, vec_dim>::calcResJac()
{
	double exp_lambda{};
	int row_idx;
	
	double Z;
	double int1[vec_dim] = {0};
	double int2[vec_dim*vec_dim] = {0};
	double int3[vec_dim*vec_dim] = {0};
	double int4[vec_dim] = {0};
	
	int i[vec_dim] = {0, 0, 0, 1, 1};
	int j[vec_dim] = {0, 1, 2, 1, 2};
	double delta[vec_dim] = {1, 0, 0, 1, 0};
	
	#pragma unroll
	for (int coord_idx = 0; coord_idx < order; ++coord_idx)
	{
		row_idx = N_COORDS*coord_idx;
		exp_lambda = calcExpLambda(row_idx);
		
		Z += exp_lambda * lebedev_weights[coord_idx];
		
		#pragma unroll
		for (int m = 0; m < vec_dim; ++m)
		{
			int1[m] += calcInt1(exp_lambda, coord_idx, row_idx, i[m], j[m]);
			int4[m] += calcInt4(exp_lambda, coord_idx, row_idx, i[m], j[m]);
			
			#pragma unroll
			for (int n = 0; n < vec_dim; ++n)
			{
				int2[vec_dim*m + n] += calcInt2(exp_lambda, coord_idx, row_idx,
						   	   	   	   	   	    i[m], j[m], i[n], j[n]);
				int3[vec_dim*m + n] += calcInt3(exp_lambda, coord_idx, row_idx,
   	   	   	   	   	    					    i[m], j[m], i[n], j[n]);
			}
		}
	}
	
	#pragma unroll
	for (int m = 0; m < vec_dim; ++m)
	{
		Res[m] = int1[m] / Z 
				 - (1.0 / 3.0) * delta[m]
				 - Q[m];
		
		#pragma unroll
		for (int n = 0; n < vec_dim; ++n)
		{
			if (n == 0 || n == 3)
				Jac(m, n) = int3[vec_dim*m + n] / Z
							- int1[m] * int4[n] / (Z*Z);
			else
				Jac(m, n) = 2 * int2[vec_dim*m + n] / Z
					    	- 2 * int1[m] * int1[n] / (Z*Z);
		}
	}
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcExpLambda(int row_idx)
{
	double lambda_xi_xi{0.0};
	
	// calculate contribution from upper off-diagonal elements
	lambda_xi_xi += Lambda[1] 		 
				    * lebedev_coords[row_idx]
				    * lebedev_coords[row_idx + 1];
	lambda_xi_xi += Lambda[2]
				    * lebedev_coords[row_idx]
				    * lebedev_coords[row_idx + 2];
	lambda_xi_xi += Lambda[4]
				    * lebedev_coords[row_idx + 1]
				    * lebedev_coords[row_idx + 2];
	
	// x2 to get lower off-diagonal elements
	lambda_xi_xi *= 2;
	
	// calculate contribution from diagonal elements
	lambda_xi_xi += Lambda[0]
				    * (lebedev_coords[row_idx]
				  	   * lebedev_coords[row_idx]
				  	   -
					   lebedev_coords[row_idx + 2]
				  	   * lebedev_coords[row_idx + 2]);
	lambda_xi_xi += Lambda[3]
				    * (lebedev_coords[row_idx + 1]
					   * lebedev_coords[row_idx + 1]
				       -
					   lebedev_coords[row_idx + 2]
					   * lebedev_coords[row_idx + 2]);
	
	return exp(lambda_xi_xi);
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt1
(const double exp_lambda, const int coord_idx, 
 const int row_idx, const int i_m, const int j_m)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * lebedev_coords[row_idx + i_m]
		   * lebedev_coords[row_idx + j_m];
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt2
(const double exp_lambda, const int coord_idx, const int row_idx,
const int i_m, const int j_m, const int i_n, const int j_n)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * lebedev_coords[row_idx + i_m]
		   * lebedev_coords[row_idx + j_m]
		   * lebedev_coords[row_idx + i_n]
		   * lebedev_coords[row_idx + j_n];
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt3
(const double exp_lambda, const int coord_idx, const int row_idx,
const int i_m, const int j_m, const int i_n, const int j_n)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * lebedev_coords[row_idx + i_m]
		   * lebedev_coords[row_idx + j_m]
		   * (lebedev_coords[row_idx + i_n]
			  * lebedev_coords[row_idx + i_n]
			  - 
			  lebedev_coords[row_idx + 2]
			  * lebedev_coords[row_idx + 2]);
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt4
(const double exp_lambda, const int coord_idx, 
const int row_idx, const int i_m, const int j_m)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * (lebedev_coords[row_idx + i_m]
			  * lebedev_coords[row_idx + i_m]
			  - 
			  lebedev_coords[row_idx + 2]
			  * lebedev_coords[row_idx + 2]);
}

#endif