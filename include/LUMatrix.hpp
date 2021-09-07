#ifndef LU_MATRIX_HPP
#define LU_MATRIX_HPP

#include <iostream>



/**
 * \brief Identical class to LUMatrixGPU, except this class is a pure C++ class
 * (rather than a Cuda class) so that it only runs on the CPU. See LUMatrixGPU
 * for more detailed documentation and a usage example.
 */
template<typename T, unsigned int N>
class LUMatrix
{
public:
	LUMatrix();
	LUMatrix(T*);
	~LUMatrix();
	const T& operator() (const unsigned int i, const unsigned int j) const;
	T& operator() (const unsigned int i, const unsigned int j);
	void copy(T*);
	void compute_lu_factorization();
	void solve(T* x);

private:
	// data stored row major
	T A[N*N];
	unsigned int row_index[N];
};

template<typename T, unsigned int N>
std::ostream& operator<< (std::ostream& os, const LUMatrix<T, N>& mat);

#endif
