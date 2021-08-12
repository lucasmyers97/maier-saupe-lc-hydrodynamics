#include "LUMatrixGPU.hpp"
#include <math.h>
#include <iostream>
#include <boost/preprocessor/array/elem.hpp>

#define Q_ROW (5, (0, 0, 0, 1, 1))
#define Q_COL (5, (0, 1, 2, 1, 2))
#define DELTA_VEC (5, (1, 0, 0, 1, 0))

template <typename T, int order, unsigned int vec_dim>
class LagrangeMultiplierGPU
{
public:
	__device__ LagrangeMultiplierGPU(T* in_lebedev_coords, 
								     T* in_lebedev_weights)
	: lebedev_coords(in_lebedev_coords)
	, lebedev_weights(in_lebedev_weights)
	{};
	
	LUMatrixGPU<T, vec_dim> Jac;
	T Res[vec_dim];
	__device__ inline void initializeInversion(const T* Q_in);

private:
	__device__ inline void calcResJac();
	__device__ inline void calcQ();
	__device__ inline void factorJac();
	__device__ inline void calcdLambda();
	__device__ inline void updateLambda();


	T* lebedev_coords;
	T* lebedev_weights;

	T Q[vec_dim];
	T Lambda[vec_dim];
	T dLambda[vec_dim];

};



template<typename T, int order, unsigned int vec_dim>
__device__ inline
void 
LagrangeMultiplierGPU<T, order, vec_dim>::initializeInversion(const T* Q_in)
{
	for (int i = 0; i < vec_dim; ++i)
	{
		Q[i] = Q_in[i];
		Res[i] = Q[i]; // can explicitly compute for Lambda = 0
	}
	
	for (int i = 0; i < vec_dim; ++i)
		for (int j = 0; j < vec_dim; ++j)
		{
			if (i == j)
				Jac(i, j) = 2.0 / 15.0;
			else
				Jac(i, j) = 0;
		}
}

__global__
void initializeLagrangeMultiplier(const double *Q_in, double *Res, double *Jac)
{
	extern __shared__ LagrangeMultiplierGPU<double, 590, 5> lm[];
	
	lm[0].initializeInversion(Q_in);
	for (int i = 0; i < 5; ++i)
	{
		Res[i] = lm[0].Res[i];
		for (int j = 0; j < 5; ++j)
			Jac[5*i + j] = lm[0].Jac(i, j);
	}
}

int main()
{
	constexpr int vec_dim = 5;
	double Q[vec_dim] = {2.0 / 3.0, 0, 0, -1.0/3.0, 0};
	double *d_Q, *d_Res, *d_Jac;
	
	cudaMalloc(&d_Q, vec_dim*sizeof(double));
	cudaMalloc(&d_Res, vec_dim*sizeof(double));
	cudaMalloc(&d_Jac, vec_dim*vec_dim*sizeof(double));
	cudaMemcpy(d_Q, Q, vec_dim*sizeof(double), cudaMemcpyHostToDevice);
	
	initializeLagrangeMultiplier
		<<<1, 1, sizeof(LagrangeMultiplierGPU<double, 590, vec_dim>)>>>
		(d_Q, d_Res, d_Jac);
	
	double Res[vec_dim] = {};
	double Jac[vec_dim*vec_dim] = {};
	cudaMemcpy(Res, d_Res, vec_dim*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Jac, d_Jac,
			   vec_dim*vec_dim*sizeof(double), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < vec_dim; ++i)
		std::cout << Res[i] << std::endl;
	
	for (int i = 0; i < vec_dim; ++i)
	{
		for (int j = 0; j < vec_dim; ++j)
			std::cout << Jac[vec_dim*i + j] << " ";
		std::cout << std::endl;
	}
	
	return 0;
}