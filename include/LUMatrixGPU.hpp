#ifndef LU_MATRIX_GPU_HPP
#define LU_MATRIX_GPU_HPP

#include <iostream>

template<typename T, unsigned int N>
class LUMatrixGPU
{
public:
	__host__ __device__ LUMatrixGPU();
	__host__ __device__ LUMatrixGPU(T*);
	__host__ __device__ ~LUMatrixGPU();
	__host__ __device__ const T& operator() (const unsigned int i, const unsigned int j) const;
	__host__ __device__ T& operator() (const unsigned int i, const unsigned int j);
	__host__ __device__ void copy(T*);
	__host__ __device__ void compute_lu_factorization();
	__host__ __device__ void solve(T* x);

private:
	// data stored row major
	T A[N*N];
	unsigned int row_index[N];
};

template<typename T, unsigned int N>
__host__
std::ostream& operator<< (std::ostream& os, const LUMatrixGPU<T, N>& mat);

#endif
