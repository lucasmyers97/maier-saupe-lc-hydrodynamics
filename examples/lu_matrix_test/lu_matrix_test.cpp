#include "LU_Matrix.hpp"
#include <stdlib.h>

int main()
{
	// Decide dimensions
	const unsigned int N{10};

	// Generate LU_Matrix from array
	double* mat = new double[N*N];
	for (unsigned int i = 0; i < N*N; i++)
	{
		mat[i] = rand() / double(RAND_MAX);
	}
	LU_Matrix<double, N> lu_mat(mat);
	LU_Matrix<double, N> orig_mat(mat);
	delete[] mat;
	std::cout << lu_mat << std::endl;

	// Compute the factorization
	lu_mat.compute_lu_factorization();
	std::cout << lu_mat << std::endl;

	// Decompose lower and upper part of matrices
	LU_Matrix<double, N> l_mat;
	LU_Matrix<double, N> u_mat;
	for (unsigned int i = 0; i < N; ++i)
		for (unsigned int j = 0; j < N; ++j)
			if (i > j) { l_mat(i, j) = lu_mat(i, j); }
			else if (i < j) { u_mat(i, j) = lu_mat(i, j); }
			else if (i == j)
			{
				l_mat(i, j) = 1;
				u_mat(i, j) = lu_mat(i, j);
			}
	std::cout << l_mat << std::endl;
	std::cout << u_mat << std::endl;

	// Multiply out matrices again, check to see if they're the same as before
	LU_Matrix<double, N> sol_mat;
	for (unsigned int i = 0; i < N; ++i)
		for (unsigned int j = 0; j < N; ++j)
			for (unsigned int k = 0; k < N; ++k)
				sol_mat(i, j) += l_mat(i, k)*u_mat(k, j);
	std::cout << sol_mat << std::endl;

	double *b = new double[N];
	double *x = new double[N];
	for (unsigned int i = 0; i < N; ++i)
	{
		b[i] = rand() / double(RAND_MAX);
		std::cout << b[i] << std::endl;
		x[i] = b[i];
	}
	std::cout << std::endl;

	lu_mat.solve(x);

	for (unsigned int i = 0; i < N; ++i)
	{
		b[i] = 0;
		for (unsigned int k = 0; k < N; ++k)
			b[i] += orig_mat(i, k)*x[k];
		std::cout << b[i] << std::endl;
	}
	std::cout << std::endl;
	std::cout << sizeof(LU_Matrix<double, N>) << std::endl;
	std::cout << sizeof(double) << std::endl;

	return 0;
}
