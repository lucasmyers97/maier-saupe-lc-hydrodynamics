#include <math.h>
#include "LUMatrixGPU.hpp"

#define N_COORDS 3

template <typename T, int order, unsigned int vec_dim>
class LagrangeMultiplierGPU
{
public:
	__device__ LagrangeMultiplierGPU() {};

private:
	__device__ inline void calcQ();
	__device__ inline void factorJac();
	__device__ inline void calcdLambda();
	__device__ inline void updateLambda();
	__device__ inline void readLebedevGlobal(T* g_lebedev_coords,
											 T* g_lebedev_weights,
											 T* s_lebedev_coords,
											 T* s_lebedev_weights,
											 int t_idx, int n_threads);
	__device__ inline void setLebedevData(T* in_lebedev_coords,
										  T* in_lebedev_weights);
	__device__ inline void initializeInversion(const T* Q_in);
	__device__ inline void calcResJac();
	__device__ inline double calcExpLambda(int row_idx);
	__device__ inline double calcInt1(double exp_lambda,
									  int coord_idx, int row_idx,
			   	   	   	   	   	   	  int i_m, int j_m);
	__device__ inline double calcInt2(double exp_lambda,
									  int coord_idx, int row_idx,
			   	   	   	   	   	      int i_m, int j_m,
									  int i_n, int j_n);
	__device__ inline double calcInt3(double exp_lambda,
									  int coord_idx, int row_idx,
				   	   	   	   	   	  int i_m, int j_m,
									  int i_n, int j_n);
	__device__ inline double calcInt4(double exp_lamda,
									  int coord_idx, int row_idx,
					   	   	   	   	  int i_m, int j_m);


	T* lebedev_coords{NULL};
	T* lebedev_weights{NULL};

	LUMatrixGPU<T, vec_dim> Jac;
	T Res[vec_dim];
	T Q[vec_dim];
	T Lambda[vec_dim];
	T dLambda[vec_dim];

};



template <typename T, int order, unsigned int vec_dim>
__device__ inline
void
LagrangeMultiplierGPU<T, order, vec_dim>::readLebedevGlobal
(T* g_lebedev_coords, T* g_lebedev_weights,
 T* s_lebedev_coords, T* s_lebedev_weights,
 int t_idx, int n_threads)
{
	#pragma unroll
	for (int i = t_idx; i < order; i += n_threads)
	{
		s_lebedev_weights[i] = g_lebedev_weights[i];
	}

	#pragma unroll
	for (int i = t_idx; i < N_COORDS; ++i)
	{
		s_lebedev_coords[i] = g_lebedev_coords[i];
	}
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline void
LagrangeMultiplierGPU<T, order, vec_dim>::setLebedevData(T* in_lebedev_coords,
													     T* in_lebedev_weights)
{
	lebedev_coords = in_lebedev_coords;
	lebedev_weights = in_lebedev_weights;
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
	double int1[vec_dim];
	double int2[vec_dim*vec_dim];
	double int3[vec_dim*vec_dim];
	double int4[vec_dim];
	
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
			if (n == 1 || n == 4)
				Jac(m, n) = 2 * int2[vec_dim*m + n] / Z
					    	- 2 * int1[m] * int1[n] / (Z*Z);
			else
				Jac(m, n) = int3[vec_dim*m + n] / Z
							- int1[m] * int4[n] / (Z*Z);
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
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt1(double exp_lambda,
												   int coord_idx, int row_idx,
												   int i_m, int j_m)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * lebedev_coords[row_idx + i_m]
		   * lebedev_coords[row_idx + j_m];
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt2(double exp_lambda,
												   int coord_idx, int row_idx,
												   int i_m, int j_m,
												   int i_n, int j_n)
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
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt3(double exp_lambda,
												   int coord_idx, int row_idx,
												   int i_m, int j_m,
												   int i_n, int j_n)
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
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt4(double exp_lambda,
												   int coord_idx, int row_idx,
												   int i_m, int j_m)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * (lebedev_coords[row_idx + i_m]
			  * lebedev_coords[row_idx + i_m]
			  - 
			  lebedev_coords[row_idx + 2]
			  * lebedev_coords[row_idx + 2]);
}
