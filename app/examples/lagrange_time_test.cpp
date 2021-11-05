#include <iostream>
#include <vector>
#include <deal.II/lac/vector.h>
#include "LagrangeMultiplier.hpp"
#include <chrono>

int main()
{
	unsigned int num_pts{20};
	dealii::Vector<double> x(num_pts);
	dealii::Vector<double> y(num_pts);

	double left = -5.0;
	double right = 5.0;
	double space = (right - left) / (num_pts - 1.0);
	for (unsigned int i = 0; i < num_pts; ++i)
	{
		x[i] = i*space + left;
		y[i] = x[i];
	}

	unsigned int vec_dim{5};
	std::vector<std::vector<dealii::Vector<double>>>
	Q(num_pts,
	  std::vector<dealii::Vector<double>>(num_pts, dealii::Vector<double>(vec_dim)));
	double S{0.675};
	for (unsigned int i = 0; i < num_pts; ++i)
	{
		for (unsigned int j = 0; j < num_pts; ++j)
		{
			double phi{std::atan2(y[i], x[i])};
			phi += 2.0*M_PI;
			phi = std::fmod(phi, 2.0*M_PI);

			Q[i][j][0] = S * (std::cos(phi / 2.0)*std::cos(phi / 2.0) - 1.0 / 3.0);
			Q[i][j][1] = S * std::cos(phi / 2.0)*std::sin(phi / 2.0);
			Q[i][j][2] = 0.0;
			Q[i][j][3] = S * (std::sin(phi / 2.0)*std::sin(phi / 2.0) - 1.0 / 3.0);
			Q[i][j][4] = 0.0;
		}
	}

	const int order{590};
	double lagrange_alpha{1.0};
	double tol{1e-8};
	int max_iter{20};

	dealii::Vector<double> Lambda(vec_dim);
	LagrangeMultiplier<order> lagrange_multiplier(lagrange_alpha,
												  tol, max_iter);

	auto start = std::chrono::high_resolution_clock::now();
	for (unsigned int i = 0; i < num_pts; ++i)
	{
		for (unsigned int j = 0; j < num_pts; ++j)
		{
			lagrange_multiplier.setQ(Q[i][j]);
			lagrange_multiplier.returnLambda(Lambda);
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// 18733 milliseconds for Debug (400 inversions)
	// 1503 milliseconds for Release (400 inversions)
	std::cout << duration.count()
			  << " milliseconds for "
			  << num_pts*num_pts
			  << " inversions" <<std::endl;

	return 0;
}
