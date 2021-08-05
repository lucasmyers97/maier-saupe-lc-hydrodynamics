#include "LU_Matrix.hpp"
#include <iostream>
#include <iomanip>

template<typename T, unsigned int N>
LU_Matrix<T, N>::LU_Matrix()
{
	A = new T[N*N]{};
	row_index = new unsigned int[N]{};
}



template<typename T, unsigned int N>
LU_Matrix<T, N>::LU_Matrix(T* input_data)
{
	A = new T[N*N];
	for (unsigned int i = 0; i < N*N; ++i)
		A[i] = input_data[i];
	
	row_index = new unsigned int[N]{};
}



template<typename T, unsigned int N>
LU_Matrix<T, N>::~LU_Matrix()
{
	delete[] A;
	delete[] row_index;
}



template<typename T, unsigned int N>
inline
const T& LU_Matrix<T, N>::operator() (unsigned int i, unsigned int j) const
{
	// Stored row major
	return A[N*i + j];
}



template<typename T, unsigned int N>
inline
T& LU_Matrix<T, N>::operator() (unsigned int i, unsigned int j)
{
	// Stored row major
	return A[N*i + j];
}



template<typename T, unsigned int N>
void LU_Matrix<T, N>::copy(T* input_data)
{
	for (unsigned int i = 0; i < N*N; ++i)
		A[i] = input_data[i];
}



template<typename T, unsigned int N>
void LU_Matrix<T, N>::compute_lu_factorization()
{
	unsigned int i, j, k, imax;
	double max_element, temp_element;
	double *row_scale = new double[N];
	
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
void LU_Matrix<T, N>::solve(T* b)
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



template<typename T, unsigned int N>
std::ostream& operator<< (std::ostream& os, const LU_Matrix<T, N>& mat)
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

#include "LU_Matrix.inst"
