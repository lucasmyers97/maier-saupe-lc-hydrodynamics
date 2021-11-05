#ifndef LU_MATRIX_GPU_HPP
#define LU_MATRIX_GPU_HPP

#include <iostream>
#include <iomanip>



/**
 * \brief Holds a fixed-size NxN matrix which can live on either the device or
 * host side. Has methods to LU decompose, as well as solve a matrix equation
 * given some input vector.
 *
 * A typical use case:
 * @code
 *  #include <stdlib.h>
 *  #include "LUMatrixGPU.hpp"
 *
 * constexpr int N = 5;
 *
 * 	// Generate an empty LUMatrixGPU object, populate randomly
 *  LUMatrixGPU<double, N> lu_mat;
 *  for (unsigned int i = 0; i < N; ++i)
 *		for (unsigned int j = 0; j < N; ++j)
 *			lu_mat(i, j) = rand() / double(RAND_MAX);
 *
 * 	// Generate right-hand-sides
 *  double *b = new double[N];
 *  for (unsigned int i = 0; i < N; ++i)
 *  {
 * 		b[N*n + i] = rand() / double(RAND_MAX);
 * 	}
 *
 *  lu_mat.compute_lu_factorization();
 *  lu_mat.solve(b);
 * @endcode
 * Here we populate the LU matrix by using the () operator -- the entries are
 * just randomly generated. We also randomly generate a right-hand-side vector
 * to the matrix equation \f$Ax = b\f$. Then we factorize the matrix, and solve
 * the matrix equation. The solution is put back into the `b` array.
 *
 * Note that this object can also be used, in an identical way, in a kernel,
 * given that all of the methods are both `__device__`- and `__host__`-side
 * methods.
 */
template<typename T, unsigned int N>
class LUMatrixGPU
{
public:

	/**
	 * \brief This default constructor is empty -- the arrays are initialized to
	 * 0 by default.
	 */
	__host__ __device__ LUMatrixGPU() {};

	/**
	 * \brief This constructor copies the matrix entries from an NxN array of
	 * type T.
	 */
	__host__ __device__ LUMatrixGPU(T*);

	/**
	 * \brief Used to index the matrix object. Can only read entries, as a
	 * `const` function.
	 *
	 * @param[in] i Row index of the entry.
	 * @param[in] j Column index of the entry.
	 *
	 * @return Returns (i, j) entry of the matrix.
	 */
	__host__ __device__ const T& operator() (const unsigned int i, 
											 const unsigned int j) const;

	/**
	 * \brief Used to index the matrix object. Can read entries and write
	 * entries to matrix.
	 *
	 * @param[in] i Row index of the entry.
	 * @param[in] j Column index of the entry.
	 *
	 * @return Returns (i, j) entry of the matrix.
	 */
	__host__ __device__ T& operator() (const unsigned int i, 
									   const unsigned int j);

	/**
	 * \brief Copies array of type T and length NxN to the matrix.
	 */
	__host__ __device__ void copy(T*);

	/**
	 * \brief Computes LU factorization of the matrix, writes back factorization
	 * to the matrix.
	 *
	 * LU-factorization is a process by which a matrix \f$A\f$ is rewritten as
	 * a product of a lower triangular matrix \f$L\f$ and an upper triangular
	 * matrix \f$U\f$, so that \f$LU = A\f$. Note that in this algorithm, the 
	 * factorized matrix is written back to the original matrix:
	 * The U matrix is written to the upper triangular portion (including the
	 * diagonal), and the L matrix is written to the lower triangular portion
	 * with the diagonal values implicitly understood to be all one's. 
	 *
	 * @see <a href="http://numerical.recipes/">Numerical Recipes</a> section 
	 * 2.3. 
	 */
	__host__ __device__ void compute_lu_factorization();

	/**
	 * \brief Computes the solution to the equation \f$Ax = b\f$ given that the
	 * matrix \f$A\f$ has already been LU-factorized by 
	 * the compute_lu_factorization() method. The solution is written back into
	 * the right-hand-side.
	 *
	 * @param[in, out] x Array of size N and type T holding the right-hand side
	 * of the equation \f$Ax = b\f$. The solution \f$x\f$ is then written back
	 * into this array.
	 */
	__host__ __device__ void solve(T* x);

private:

	/** \brief Array holding the matrix entries. */
	// data stored row major
	T A[N*N] = {};
	/** \brief Used to keep track of permutation of LU-decomposition */
	unsigned int row_index[N] = {};
};



template<typename T, unsigned int N>
inline __host__ __device__
LUMatrixGPU<T, N>::LUMatrixGPU(T* input_data)
{
	// copy all entries from input
	for (unsigned int i = 0; i < N*N; ++i)
		A[i] = input_data[i];
}



template<typename T, unsigned int N>
inline __host__ __device__
const T& LUMatrixGPU<T, N>::operator() (unsigned int i, unsigned int j) const
{
	// Stored row major, accesses (i, j)th entry
	return A[N*i + j];
}



template<typename T, unsigned int N>
inline __host__ __device__
T& LUMatrixGPU<T, N>::operator() (unsigned int i, unsigned int j)
{
	// Stored row major, accesses (i, j)th entry
	return A[N*i + j];
}



template<typename T, unsigned int N>
inline __host__ __device__
void LUMatrixGPU<T, N>::copy(T* input_data)
{
	for (unsigned int i = 0; i < N*N; ++i)
		A[i] = input_data[i];
}



template<typename T, unsigned int N>
inline __host__ __device__
void LUMatrixGPU<T, N>::compute_lu_factorization()
{
	unsigned int i, j, k, imax;
	double max_element, temp_element;
	double row_scale[N] = {0};
	
	// Find max elements in each row for scaling
	for (i = 0; i < N; ++i)
	{
		max_element = 0;
		for (j = 0; j < N; ++j)
		{
			if (A[N*i + j] > max_element)
				max_element = A[N*i + j];
		}
		row_scale[i] = max_element;
	}
	
	for (k = 0; k < N; ++k)
	{
		// Find max element below k in current column
		max_element = A[N*k + k];
		imax = k;
		for (i = k + 1; i < N; ++i)
		{
			if (A[N*i + k] / row_scale[i] > max_element)
			{
				max_element = A[N*i + k];
				imax = i;
			}
		}
		
		// Promote max_element row, demote k-row
		if (imax != k)
		{
			for (j = 0; j < N; ++j)
			{
				temp_element = A[N*k + j];
				A[N*k + j] = A[N*imax + j];
				A[N*imax + j] = temp_element;
			}
			// won't use <= k scale factors again, don't need to switch
			row_scale[imax] = row_scale[k];
		}
		// Record in row_index array
		row_index[k] = imax;
		
		// finish i > k, j = k elements
		// add another term to i > k, j > k elements
		for (i = k + 1; i < N; ++i)
		{
			// divide by pivot elements in column k for i > k
			A[N*i + k] /= A[N*k + k];
			for (j = k + 1; j < N; ++j)
			{
				// add element to the i > k, j > k submatrix
				A[N*i + j] -= A[N*i + k]*A[N*k + j];
			}
		}
	}
}



template<typename T, unsigned int N>
inline __host__ __device__
void LUMatrixGPU<T, N>::solve(T* b)
{
	double temp_val{0};

	// Solve Ly = b, store back in b
	for (unsigned int i = 0; i < N; ++i)
	{
		// Need to permute rhs to match lu-decomposition
		if (i != row_index[i])
		{
			temp_val = b[i];
			b[i] = b[row_index[i]];
			b[row_index[i]] = temp_val;
		}
		for (unsigned int j = 0; j < i; ++j)
			b[i] -= A[N*i + j] * b[j];
	}
	
	// Solve Ux = y, store back in b
	for (int i = N - 1; i >= 0; --i)
	{
		for (int j = N - 1; j > i; --j)
			b[i] -= A[N*i + j]*b[j];
		b[i] /= A[N*i + i];
	}
}


/**
 * \brief Output the matrix in block form.
 *
 * @param os Stream to which we write the matrix
 * @param mat Matrix which is being written to a stream
 *
 * @return New stream which includes the matrix output
 */
template<typename T, unsigned int N>
__host__
std::ostream& operator<< (std::ostream& os, const LUMatrixGPU<T, N>& mat)
{
	int num_digits{3};
	int width{12};
	for (unsigned int i = 0; i < N; ++i)
	{
		for (unsigned int j = 0; j < N; ++j)
		{
			os << std::setprecision(num_digits)
			   << std::setw(width) << std::left << mat(i, j);
		}
		os << std::endl;
	}
	return os;
}

#endif
