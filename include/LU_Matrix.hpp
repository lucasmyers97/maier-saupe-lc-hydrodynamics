#ifndef LU_MATRIX_HPP
#define LU_MATRIX_HPP

#include <iostream>

template<typename T, unsigned int N>
class LU_Matrix
{
public:
	LU_Matrix();
	LU_Matrix(T*);
	~LU_Matrix();
	const T& operator() (const unsigned int i, const unsigned int j) const;
	T& operator() (const unsigned int i, const unsigned int j);
	void copy(T*);
	void compute_lu_factorization();
	void solve(T* x);

private:
	// data stored row major
	T *A;
	unsigned int *row_index;
};

template<typename T, unsigned int N>
std::ostream& operator<< (std::ostream& os, const LU_Matrix<T, N>& mat);

#endif