#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <stdlib.h>
#include "LUMatrixGPU.hpp"

namespace utf = boost::unit_test;



template <typename T, unsigned int N>
__global__
void matrix_inverse(LUMatrixGPU<T, N> *mat_array, T *b, unsigned int num_mats)
{
	extern __shared__ LUMatrixGPU<T, N> lu_mat[];
	int thread_idx = threadIdx.x;
	int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (global_idx < num_mats)
	{
        lu_mat[thread_idx] = mat_array[global_idx];
		lu_mat[thread_idx].compute_lu_factorization();
		lu_mat[thread_idx].solve(&b[N*thread_idx]);
	}
}



BOOST_AUTO_TEST_CASE(lu_matrix_gpu_inversion_test, *utf::tolerance(1e-12))
{
	// Decide dimensions, number of matrices
	const unsigned int N{3};
	const unsigned int num_mats{10};

	// Generate num_mats number of random LUMatrixGPUs
	LUMatrixGPU<double, N>* lu_mat = new LUMatrixGPU<double, N>[num_mats];
	for (unsigned int n = 0; n < num_mats; ++n)
		for (unsigned int i = 0; i < N; ++i)
			for (unsigned int j = 0; j < N; ++j)
				lu_mat[n](i, j) = rand() / double(RAND_MAX);
				
	// Generate num_mats number of rhs's
	double *b = new double[N*num_mats];
	for (unsigned int n = 0; n < num_mats; ++n)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			b[N*n + i] = rand() / double(RAND_MAX);
		}
	}
				
    // Copy matrices/rhs's over to device
	LUMatrixGPU<double, N> *d_lu_mat;
	double *d_b;
	cudaMalloc(&d_lu_mat, num_mats*sizeof(LUMatrixGPU<double, N>));	
	cudaMalloc(&d_b, num_mats*N*sizeof(double));
	cudaMemcpy(d_lu_mat, lu_mat, 
			   num_mats*sizeof(LUMatrixGPU<double, N>), 
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, num_mats*N*sizeof(double),
		       cudaMemcpyHostToDevice);
			   
	matrix_inverse
        <<<1, num_mats, num_mats*sizeof(LUMatrixGPU<double, N>)>>>
        (d_lu_mat, d_b, num_mats);
	
    double *out_b = new double[N*num_mats];
	cudaError_t error = cudaMemcpy(out_b, d_b, num_mats*N*sizeof(double),
	           				       cudaMemcpyDeviceToHost);
	
	double entry{0.0};
	for (unsigned int n = 0; n < num_mats; ++n)
	{
		for (unsigned int i = 0; i < N; ++i)
		{
			entry = 0;
			for (unsigned int k = 0; k < N; ++k)
				entry += lu_mat[n](i, k)*out_b[N*n + k];
			
            BOOST_TEST(entry == b[N*n + i]);
		}
	}
	
	cudaFree(d_lu_mat);
	cudaFree(d_b);
	delete[] lu_mat;
	delete[] b;
    delete[] out_b;
}