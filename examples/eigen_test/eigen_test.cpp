// This script creates a matrix m, and a vector v
// then solves the linear system m*x = v with a QR
// decomposition with column pivoting algorithm.
// This is all done using the package Eigen.

#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

int main()
{
	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2;
	m(0, 1) = 1;
	m(1, 1) = m(0, 0) + m(1, 0);

	std::cout << "Here is the matrix:\n" << m << std::endl;

	VectorXd v(2);
	v(0) = 3;
	v(1) = 1;

	std::cout << "Here is the vector:\n" << v << std::endl;

	VectorXd x(2);
	x = m.colPivHouseholderQr().solve(v);

	std::cout << "Here is the solution:\n" << x << std::endl;

	std::cout << "Here is m*x:\n" << m*x << std::endl;

	return 0;
}
