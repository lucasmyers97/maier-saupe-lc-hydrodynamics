#include "LUMatrixGPU.hpp"

template <typename T, int order, unsigned int vec_dim>
class LagrangeMultiplierGPU
{
public:
	LagrangeMultiplierGPU()
	{};

private:
	void setQ(const T*);
	void calcResJac();
	void calcQ();
	void factorJac();
	void calcdLambda();
	void updateLambda();


	T* lebedev_coords;
	T* lebedev_weights;

	LUMatrixGPU<T, vec_dim> Jac;
	T Res[vec_dim];
	T Q[vec_dim];
	T Lambda[vec_dim];
	T dLambda[vec_dim];

};
