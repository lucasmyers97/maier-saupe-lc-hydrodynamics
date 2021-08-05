#include "LU_Matrix.hpp"

int main()
{
	const unsigned int N{3};
	double* mat = new double[N*N];
	for (unsigned int i = 0; i < N*N; i++)
	{
		mat[i] = (i + 1) / 3.0;
	}

	LU_Matrix<double, N> lu_mat(mat);
	delete[] mat;

	std::cout << lu_mat << std::endl;

	for (unsigned int i = 0; i < N; ++i)
		for (unsigned int j = 0; j < N; ++j)
			lu_mat(i, j) = i*j;

	std::cout << lu_mat << std::endl;

	return 0;
}
