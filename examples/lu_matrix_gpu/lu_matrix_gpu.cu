#include <iostream>
#include <stdlib.h>
#include "LUMatrixGPU.hpp"

template <typename T, unsigned int N>
__global__
void matrix_inverse(LUMatrixGPU<T, N> *mat_array, T *b)
{
//	extern __shared__ LUMatrixGPU<T, N> lu_mat[];
	int thread_idx = threadIdx.x;
	int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
//	lu_mat[thread_idx] = mat_array[global_idx];
//	lu_mat[thread_idx].solve(&b[thread_idx]);
	if (thread_idx == 0)
	{
//		lu_mat[thread_idx].compute_lu_factorization();
		b[0] = mat_array[global_idx].return_number();
	}
}

int main()
{
	// Decide dimensions, number of matrices
	const unsigned int N{10};
	const unsigned int num_mats{1};

	// Generate LUMatrixGPU
	LUMatrixGPU<double, N>* lu_mat = new LUMatrixGPU<double, N>[num_mats];
	for (unsigned int n = 0; n < num_mats; ++n)
		for (unsigned int i = 0; i < N; ++i)
			for (unsigned int j = 0; j < N; ++j)
				lu_mat[n](i, j) = rand() / double(RAND_MAX);
				
	// Generate rhs
	double *b = new double[N*num_mats];
	for (unsigned int n = 0; n < num_mats; ++n)
		for (unsigned int i = 0; i < N; ++i)
		{
			b[num_mats*n + i] = rand() / double(RAND_MAX);
			std::cout << b[num_mats*n + i] << std::endl;
		}
	std::cout << std::endl;
				
	LUMatrixGPU<double, N> *d_lu_mat;
	double *d_b;
	cudaMalloc(&d_lu_mat, num_mats*sizeof(LUMatrixGPU<double, N>));	
	cudaMalloc(&d_b, num_mats*N*sizeof(double));
	cudaMemcpy(d_lu_mat, lu_mat, 
			   num_mats*sizeof(LUMatrixGPU<double, N>), 
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, num_mats*N*sizeof(double),
		       cudaMemcpyHostToDevice);
			   
	matrix_inverse<<<1, num_mats, num_mats*sizeof(LUMatrixGPU<double, N>)>>>(d_lu_mat, d_b);
	
	cudaDeviceSynchronize();
	cudaError_t error = cudaPeekAtLastError();
	cudaMemcpy(b, d_b, num_mats*N*sizeof(double),
	           				cudaMemcpyDeviceToHost);
	           				
	           					   
	std::cout << error << std::endl;
	
	for (unsigned int n = 0; n < num_mats; ++n)
		for (unsigned int i = 0; i < N; ++i)
			std::cout << b[num_mats*n + i] << std::endl;
	
	cudaFree(d_lu_mat);
	cudaFree(d_b);
	delete[] lu_mat;
	delete[] b;
}