#include "LU_Matrix.hpp"
#include <iostream>
#include <iomanip>

template<typename T, unsigned int N>
LU_Matrix<T, N>::LU_Matrix()
{
	data_ = new T[N*N];
}

template<typename T, unsigned int N>
LU_Matrix<T, N>::LU_Matrix(T* input_data)
{
	data_ = new T[N*N];
	for (unsigned int i = 0; i < N*N; ++i)
		data_[i] = input_data[i];
}

template<typename T, unsigned int N>
LU_Matrix<T, N>::~LU_Matrix()
{
	delete[] data_;
}

template<typename T, unsigned int N>
const T& LU_Matrix<T, N>::operator() (unsigned int i, unsigned int j) const
{
	// Stored row major
	return data_[N*i + j];
}

template<typename T, unsigned int N>
T& LU_Matrix<T, N>::operator() (unsigned int i, unsigned int j)
{
	// Stored row major
	return data_[N*i + j];
}

template<typename T, unsigned int N>
void LU_Matrix<T, N>::copy(T* input_data)
{
	for (unsigned int i = 0; i < N*N; ++i)
		data_[i] = input_data[i];
}

template<typename T, unsigned int N>
std::ostream& operator<< (std::ostream& os, const LU_Matrix<T, N>& mat)
{
	int num_digits{7};
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