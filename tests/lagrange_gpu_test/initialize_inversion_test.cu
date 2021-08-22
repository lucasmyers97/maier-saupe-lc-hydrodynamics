#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#define private public
#include "LagrangeMultiplierGPU.hpp"

__global__
void initializeLagrangeMultiplier(const double *Q_in, double *Res, double *Jac)
{
	extern __shared__ LagrangeMultiplierGPU<double, 590, 5> lm[];

	lm[0].initializeInversion(Q_in);
	for (int i = 0; i < 5; ++i)
	{
		Res[i] = lm[0].Res[i];
		for (int j = 0; j < 5; ++j)
			Jac[5*i + j] = lm[0].Jac(i, j);
	}
}



BOOST_AUTO_TEST_CASE(initialize_inversion_test)
{
	constexpr int vec_dim = 5;
	double Q[vec_dim] = {2.0 / 3.0, 0, 0, -1.0/3.0, 0};
	double *d_Q, *d_Res, *d_Jac;

	cudaMalloc(&d_Q, vec_dim*sizeof(double));
	cudaMalloc(&d_Res, vec_dim*sizeof(double));
	cudaMalloc(&d_Jac, vec_dim*vec_dim*sizeof(double));
	cudaMemcpy(d_Q, Q, vec_dim*sizeof(double), cudaMemcpyHostToDevice);

	initializeLagrangeMultiplier
		<<<1, 1, sizeof(LagrangeMultiplierGPU<double, 590, vec_dim>)>>>
		(d_Q, d_Res, d_Jac);

	cudaFree(d_Q);

	double Res[vec_dim] = {};
	double Jac[vec_dim*vec_dim] = {};
	cudaMemcpy(Res, d_Res, vec_dim*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Jac, d_Jac,
			   vec_dim*vec_dim*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_Res);
	cudaFree(d_Jac);

	for (int i = 0; i < vec_dim; ++i)
	{
		BOOST_TEST(Res[i] == -Q[i]);
	}

	for (int i = 0; i < vec_dim; ++i)
	{
		for (int j = 0; j < vec_dim; ++j)
		{
			if (i == j)
			{
				BOOST_TEST(Jac[vec_dim*i + j] == 2.0/15.0);
			}
			else {
				BOOST_TEST(Jac[vec_dim*i + j] == 0);
			}
		}
	}

}