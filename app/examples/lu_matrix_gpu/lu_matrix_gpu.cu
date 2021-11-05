#include <iostream>
#include <stdlib.h>
#include "LUMatrixGPU.hpp"

template <typename T, unsigned int N>
__global__
void matrix_inverse(LUMatrixGPU<T, N> *mat_array, T *b, unsigned int num_mats)
{
	extern __shared__ LUMatrixGPU<T, N> lu_mat[];
	int thread_idx = threadIdx.x;
	int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
	lu_mat[thread_idx] = mat_array[global_idx];
	if (thread_idx < num_mats)
	{
		lu_mat[thread_idx].compute_lu_factorization();
		lu_mat[thread_idx].solve(&b[N*thread_idx]);
	}
}

int main()
{
	// Decide dimensions, number of matrices
	const unsigned int N{3};
	const unsigned int num_mats{10};

	// Generate LUMatrixGPU
	LUMatrixGPU<double, N>* lu_mat = new LUMatrixGPU<double, N>[num_mats];
	for (unsigned int n = 0; n < num_mats; ++n)
		for (unsigned int i = 0; i < N; ++i)
			for (unsigned int j = 0; j < N; ++j)
				lu_mat[n](i, j) = rand() / double(RAND_MAX);
				
	// Generate rhs
	double *b = new double[N*num_mats];
	for (unsigned int n = 0; n < num_mats; ++n)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			b[N*n + i] = rand() / double(RAND_MAX);
			std::cout << b[N*n + i] << std::endl;
		}
		std::cout << std::endl;
	}
				
	LUMatrixGPU<double, N> *d_lu_mat;
	double *d_b;
	cudaMalloc(&d_lu_mat, num_mats*sizeof(LUMatrixGPU<double, N>));	
	cudaMalloc(&d_b, num_mats*N*sizeof(double));
	cudaMemcpy(d_lu_mat, lu_mat, 
			   num_mats*sizeof(LUMatrixGPU<double, N>), 
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, num_mats*N*sizeof(double),
		       cudaMemcpyHostToDevice);
			   
	matrix_inverse<<<1, num_mats, num_mats*sizeof(LUMatrixGPU<double, N>)>>>(d_lu_mat, d_b, num_mats);
	
	cudaError_t error = cudaMemcpy(b, d_b, num_mats*N*sizeof(double),
	           				cudaMemcpyDeviceToHost);
	           					   
	std::cout << error << std::endl << std::endl;
	
	double entry{0.0};
	for (unsigned int n = 0; n < num_mats; ++n)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			entry = 0;
			for (unsigned int k = 0; k < N; ++k)
				entry += lu_mat[n](i, k)*b[N*n + k];
			std::cout << entry << std::endl;
		}
		std::cout << std::endl;
	}
	
	cudaFree(d_lu_mat);
	cudaFree(d_b);
	delete[] lu_mat;
	delete[] b;
}