#ifndef LAGRANGE_MULTIPLIER_GPU_HPP
#define LAGRANGE_MULTIPLIER_GPU_HPP

#include <math.h>
#include <assert.h>
#include "LUMatrixGPU.hpp"

namespace{
	constexpr int N_COORDS{3};
	constexpr double big_num{1e15};
}



/**
 * \brief This class is called within a kernel and, given a global pointer to
 * Lebedev coordinates and weights, does everything necessary to invert a
 * single Q-vector. 
 *
 * A typical use-case of this class is as follows:
 * @code
 * int t_idx = threadIdx.x;
 * double tol = 1e-12;
 * int max_iters = 12;
 * 
 * lm[t_idx].readLebedevGlobal(g_lebedev_coords, g_lebedev_weights,
 *							   t_idx, n_threads_in_block, s_lebedev_coords, 
 * 							   s_lebedev_weights);
 * lm[t_idx].setLebedevData(s_lebedev_coords, s_lebedev_weights);
 * lm[t_idx].setParams(tol, max_iters);
 * 
 * lm[t_idx].calcLambda(Q_in);
 * @endcode
 * where `g_lebedev_coords` and `g_lebedev_weights` are global device pointers
 * to the Lebedev coordinates and weights respectively; `n_threads_in_block` is
 * the number of threads running in the block; `s_lebedev_coords` and
 * `s_lebedev_weights` are shared device pointers to the Lebedev coordinates
 * and weights respectively; and `Q_in` is a global device pointer to an array
 * holding a Q-vector. In the snippet above the Lebedev coordinates and
 * weights are first read from global to local memory; then the
 * LagrangeMultiplierGPU object is pointed towards the shared memory Lebedev
 * data; then the Newton's method parameters for the inversion are set
 * (tolerance on the residual norm, as well as the maximal number of Newton
 * iterations); finally, `Q_in` is inverted and the corresponding Lambda value
 * is placed back in global device memory in `Q_in`.
 *
 * A lot of this rigamarole is so that, when calculating the spherical
 * integrals, the LagrangeMultiplierGPU object has as quick access as possible
 * to the Lebedev data -- hence the emphasis on getting everything into shared
 * memory.
 */
template <typename T, int order, unsigned int vec_dim>
class LagrangeMultiplierGPU
{
public:

	/**
	 * \brief The constructor is typically not called because this object is
	 * instantiated in shared memory of a kernel. Hence, the constructor is
	 * blank.
	 */
	__device__ LagrangeMultiplierGPU() {};

	/**
	 * \brief reads Lebedev coordinates and weights from global device memory
	 * into shared device memory on a particular block.
	 *
	 * @param[in] g_lebedev_coords pointer to global device array of Lebedev 
	 * 		  				   	   coordinates.
	 * @param[in] g_lebedev_weights pointer to global device array of Lebedev 
	 *							    weights.
	 * @param[in] t_idx index of the thread the function is running on.
	 * @param[in] n_threads_in_block number of threads in the given block.
	 * @param[in] s_lebedev_coords pointer to shared device array of Lebedev
	 * 						       coordinates in the given block.
	 * @param[in] s_lebedev_weights pointer to shared device array of Lebedev
	 *							    weights in given block.
	 */
	__device__ inline void readLebedevGlobal(const T* g_lebedev_coords,
											 const T* g_lebedev_weights,
											 const int t_idx, 
											 const int n_threads_in_block,
											 T* s_lebedev_coords,
											 T* s_lebedev_weights);
	
	/**
	 * \brief sets LagrangeMultiplierGPU object so that it will read in Lebedev
	 * data from shared memory when it does inversion calculation.
	 *
	 * @param[in] in_lebedev_coords pointer to shared device array of Lebedev 
	 * 							    coordinates.
	 * @param[in] in_lebedev_weights pointer to shared device array of Lebedev
	 *							     weights.
	 */
	__device__ inline void setLebedevData(T* in_lebedev_coords, 
									      T* in_lebedev_weights);

	/**
	 * \brief sets parameters for Newton's method that the LagrangeMultiplierGPU
	 * object will use during the inversion method.
	 *
	 * @param[in] tol tolerance for the norm of the residual in Newton's method
	 * 			  	  during the inversion. Once the residual norm is lower than
	 *			  	  this number, the algorithm stops.
	 * @param[in] max_iters maximal number of iterations of Newton's method 
	 *						before the inversion aborts. If this number of 
	 * 						iterations is reached before the residual norm is 
	 *						less than `tol`, an error is thrown in the kernel.
	 */
	__device__ inline void setParams(const T tol, const int max_iters);

	/**
	 * \brief calculates Lambda value, given an input Q-vector. Note that this
	 * method writes the Lambda values back into global device memory where the
	 * Q-vector was.
	 *
	 * @param[in, out] Q_in pointer to a global device array holding a Q-vector.
	 * 				        When the inversion is done, the value of the
	 *						corresponding Lambda is written back into the array.
	 */
	__device__ inline void calcLambda(T* Q_in);

private:

	/**
	 * \brief initializes the Newton's method inversion scheme. This involves
	 * copying the global device Q-values to the LagrangeMultiplierGPU object's
	 * internal Q-values, as well as setting default (i.e. Lambda = 0) values
	 * for the residual and Jacobian.
	 *
	 * @param[in] Q_in pointer to global device array holding the Q-vector 
	 *				   values.
	 */
	__device__ inline void initializeInversion(const T* Q_in);

	/**
	 * \brief caluclate Newton update `dLambda` given a particular residual and
	 * Jacobian. Need to have called calcResJac after applying the last Newton
	 * update for this function to make sense.
	 */
	__device__ inline void calcdLambda();

	/**
	 * \brief calculate the residual and Jacobian given a particular value of
	 * Lambda and Q.
	 */
	__device__ inline void calcResJac();

	/**
	 * \brief calculate \f$\exp(\Lambda_{kl} \xi_k \xi_l)\f$ for some particular
	 * \f$\Lambda\f$ where repeated indices are summed over. Here \f$\xi\f$ is
	 * a particular Lebedev coordinate indexed by `row_idx`.
	 *
	 * @param[in] row_idx given that `lebedev_coords` must be a one-dimensional
	 * 				      array (to fit in shared memory neatly), this index 
	 *					  refers to the place in the one-dimensional array where
	 * 					  the given lebedev point starts. So, if we were looking
	 *					  at the nth lebedev point in 3 dimensions, this would 
	 * 					  be 3*n.
	 */
	__device__ inline double calcExpLambda(int row_idx);

	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \xi_{i(m)} \xi_{j(m)} 
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ is the row index in the Q-tensor corresponding to the
	 * \f$m\f$th entry of the Q-vector, and \f$j(m)\f$ is the column index in
	 * the Q-tensor corresponding to the \f$m\f$th entry of the Q-vector. 
	 * The quadrature sum is calculated one term at a time to avoid having to
	 * calculate the same \f$\exp[\Lambda_{kl} \xi_k \xi_l]\f$ multiple times.
	 *
	 * @param exp_lambda holds the value of \f$\exp[\Lambda_{kl} \xi_k \xi_l]\f$
	 *					 for a particular \f$\Lambda\f$ and Lebedev point.
	 * @param coord_idx indexes which term in the quadrature sum we are
	 *					calculating.
	 * @param row_idx index used for accessing Lebedev coordinates. See
	 * 				  calcExpLambda() for more details.
	 * @param i_m row index in Q-tensor corresponding to m-th entry of Q-vector
	 * @param j_m column index in Q-tensor corresponding to m-th entry of
	 *			  Q-vector.
	 */
	__device__ inline double calcInt1Term(const double exp_lambda,
									  const int coord_idx, const int row_idx,
			   	   	   	   	   	   	  const int i_m, const int j_m);

	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \xi_{i(m)} \xi_{j(m)} \xi_{i(n)} \xi_{j(n)}
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ and \f$j(m)\f$ are as in calcInt1Term(). See
	 * calcInt1Term() for an explanation of parameters.
	 */
	__device__ inline double calcInt2Term(const double exp_lambda,
									  const int coord_idx, const int row_idx,
			   	   	   	   	   	      const int i_m, const int j_m,
									  const int i_n, const int j_n);

	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \xi_{i(m)} \xi_{j(m)} 
	 * \left(\xi_{i(n)}^2 -  \xi_{3}^2\right)
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ and \f$j(m)\f$ are as in calcInt1Term(). See
	 * calcInt1Term() for an explanation of parameters.
	 */
	__device__ inline double calcInt3Term(const double exp_lambda,
									  const int coord_idx, const int row_idx,
				   	   	   	   	   	  const int i_m, const int j_m,
									  const int i_n, const int j_n);

	/**
	 * \brief Calculate one term in the lebedev quadrature sum corresponding to
	 * the following integral: 
	 * \f$\int_{S^2} \left(\xi_{i(n)}^2 - \xi_{3}\right)
	 * \exp[\Lambda_{kl} \xi_k \xi_l] d\xi\f$
	 * where \f$i(m)\f$ and \f$j(m)\f$ are as in calcInt1Term(). See
	 * calcInt1Term() for an explanation of parameters.
	 */
	__device__ inline double calcInt4Term(const double exp_lamda,
									  const int coord_idx, const int row_idx,
					   	   	   	   	  const int i_m, const int j_m);

	/** \brief Pointer to Lebedev quadrature coordinates.
	 *
	 * Note that these are stored row-major, so that the x, y, z components of
	 * each point are adjacent in the array (program more efficient that way).
	*/
	T* lebedev_coords{NULL};
	/** \brief Pointer to Lebedev quadrature weights */
	T* lebedev_weights{NULL};

	/** \brief Matrix object which can be LU-decomposed, represents Jacobian */
	LUMatrixGPU<T, vec_dim> Jac;
	/** \brief Array holding residual of the inversion */
	T Res[vec_dim];
	/** \brief Array holding Q-vector */
	T Q[vec_dim];
	/** \brief Array holding current estimate of Lambda corresponding to 
	Q-vector */
	T Lambda[vec_dim] = {0};
	/** \brief Array holding Newton update for Lambda*/
	T dLambda[vec_dim];

	/** \brief Tolerance for the norm of the residual for Newton's method */
	T tol;
	/** \brief Maximum number of iterations for Newton's method */
	int max_iters;
};



template <typename T, int order, unsigned int vec_dim>
__device__ inline
void
LagrangeMultiplierGPU<T, order, vec_dim>::readLebedevGlobal
(const T* g_lebedev_coords, const T* g_lebedev_weights,
 const int t_idx, const int n_threads_in_block,
 T* s_lebedev_coords, T* s_lebedev_weights)
{
	#pragma unroll
	for (int i = t_idx; i < order; i += n_threads_in_block)
	{
		s_lebedev_weights[i] = g_lebedev_weights[i];
	}

	#pragma unroll
	for (int i = t_idx; i < N_COORDS*order; i += n_threads_in_block)
	{
		s_lebedev_coords[i] = g_lebedev_coords[i];
	}
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline void
LagrangeMultiplierGPU<T, order, vec_dim>::setLebedevData
(T* in_lebedev_coords, T* in_lebedev_weights)
{
	lebedev_coords = in_lebedev_coords;
	lebedev_weights = in_lebedev_weights;
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline 
void LagrangeMultiplierGPU<T, order, vec_dim>::setParams
(const T in_tol, const int in_max_iters)
{
	tol = in_tol;
	max_iters = in_max_iters;
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline void 
LagrangeMultiplierGPU<T, order, vec_dim>::calcLambda(T* Q_in)
{
	// Get initial residual, jacobian for given Q
	initializeInversion(Q_in);

	// Calculate norm of initial residual
	double res_norm = 0;
	for (int i = 0; i < vec_dim; ++i)
		res_norm += Res[i]*Res[i];
	res_norm = sqrt(res_norm);

	// Do Newton's method until norm < tolerance, or iterations > max iterations
	int iters = 0;
	while (iters < max_iters && res_norm > tol)
	{
		calcdLambda();
		for (int i = 0; i < vec_dim; ++i)
			Lambda[i] += dLambda[i];

		calcResJac();

		res_norm = 0;
		for (int i = 0; i < vec_dim; ++i)
			res_norm += Res[i]*Res[i];
		res_norm = sqrt(res_norm);

		++iters;
	}
	
	// Throw an error if inversion failed
	assert(res_norm < tol);

	// Copy inversion back to global device memory
	for (int i = 0; i < vec_dim; ++i)
		Q_in[i] = Lambda[i];
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline void 
LagrangeMultiplierGPU<T, order, vec_dim>::calcdLambda()
{
	Jac.compute_lu_factorization();

	// Copy -res to dLambda because solver writes solution back to rhs
	for (int i = 0; i < vec_dim; ++i)
		dLambda[i] = -Res[i];
	Jac.solve(dLambda);
}



template<typename T, int order, unsigned int vec_dim>
__device__ inline
void 
LagrangeMultiplierGPU<T, order, vec_dim>::initializeInversion(const T* Q_in)
{
	for (int i = 0; i < vec_dim; ++i)
	{
		Q[i] = Q_in[i];
		Res[i] = -Q[i]; // can explicitly compute for Lambda = 0
	}
	
	// for Jacobian, compute 2/15 on diagonal, 0 elsewhere for Lambda = 0
	for (int i = 0; i < vec_dim; ++i)
		for (int j = 0; j < vec_dim; ++j)
		{
			if (i == j)
				Jac(i, j) = 2.0 / 15.0;
			else
				Jac(i, j) = 0;
		}
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
void
LagrangeMultiplierGPU<T, order, vec_dim>::calcResJac()
{
	double exp_lambda{};
	int row_idx{};
	
	// Arrays holding all values of Lebedev quadrature
	double Z{};
	double int1[vec_dim] = {0};
	double int2[vec_dim*vec_dim] = {0};
	double int3[vec_dim*vec_dim] = {0};
	double int4[vec_dim] = {0};
	
	// Row & Col indices of Q-tensor corresponding to Q-vector entries
	int i[vec_dim] = {0, 0, 0, 1, 1};
	int j[vec_dim] = {0, 1, 2, 1, 2};

	// KroneckerDelta but in Q-vector form
	double delta[vec_dim] = {1, 0, 0, 1, 0};
	
	// Calculate each term in Lebedev quadrature for each integral, add to total
	// quadrature value until we've summed all terms
	#pragma unroll
	for (int coord_idx = 0; coord_idx < order; ++coord_idx)
	{
		row_idx = N_COORDS*coord_idx;
		exp_lambda = calcExpLambda(row_idx);
		
		Z += exp_lambda * lebedev_weights[coord_idx];
		
		#pragma unroll
		for (int m = 0; m < vec_dim; ++m)
		{
			int1[m] += calcInt1Term(exp_lambda, coord_idx, row_idx, i[m], j[m]);
			int4[m] += calcInt4Term(exp_lambda, coord_idx, row_idx, i[m], j[m]);
			
			#pragma unroll
			for (int n = 0; n < vec_dim; ++n)
			{
				int2[vec_dim*m + n] += calcInt2Term(exp_lambda, coord_idx, 
													row_idx, i[m], j[m], 
													i[n], j[n]);
				int3[vec_dim*m + n] += calcInt3Term(exp_lambda, coord_idx, 
													row_idx, i[m], j[m], 
													i[n], j[n]);
			}
		}
	}
	
	// Calculate each entry of residual and Jacobian using integral values
	#pragma unroll
	for (int m = 0; m < vec_dim; ++m)
	{
		Res[m] = int1[m] / Z 
				 - (1.0 / 3.0) * delta[m]
				 - Q[m];
		
		#pragma unroll
		for (int n = 0; n < vec_dim; ++n)
		{
			if (n == 0 || n == 3)
				Jac(m, n) = int3[vec_dim*m + n] / Z
							- int1[m] * int4[n] / (Z*Z);
			else
				Jac(m, n) = 2 * int2[vec_dim*m + n] / Z
					    	- 2 * int1[m] * int1[n] / (Z*Z);
		}
	}
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcExpLambda(int row_idx)
{
	double lambda_xi_xi{0.0};
	
	// calculate contribution from upper off-diagonal elements
	lambda_xi_xi += Lambda[1] 		 
				    * lebedev_coords[row_idx]
				    * lebedev_coords[row_idx + 1];
	lambda_xi_xi += Lambda[2]
				    * lebedev_coords[row_idx]
				    * lebedev_coords[row_idx + 2];
	lambda_xi_xi += Lambda[4]
				    * lebedev_coords[row_idx + 1]
				    * lebedev_coords[row_idx + 2];
	
	// x2 to get lower off-diagonal elements
	lambda_xi_xi *= 2;
	
	// calculate contribution from diagonal elements
	lambda_xi_xi += Lambda[0]
				    * (lebedev_coords[row_idx]
				  	   * lebedev_coords[row_idx]
				  	   -
					   lebedev_coords[row_idx + 2]
				  	   * lebedev_coords[row_idx + 2]);
	lambda_xi_xi += Lambda[3]
				    * (lebedev_coords[row_idx + 1]
					   * lebedev_coords[row_idx + 1]
				       -
					   lebedev_coords[row_idx + 2]
					   * lebedev_coords[row_idx + 2]);
	
	return exp(lambda_xi_xi);
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt1Term
(const double exp_lambda, const int coord_idx, 
 const int row_idx, const int i_m, const int j_m)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * lebedev_coords[row_idx + i_m]
		   * lebedev_coords[row_idx + j_m];
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt2Term
(const double exp_lambda, const int coord_idx, const int row_idx,
const int i_m, const int j_m, const int i_n, const int j_n)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * lebedev_coords[row_idx + i_m]
		   * lebedev_coords[row_idx + j_m]
		   * lebedev_coords[row_idx + i_n]
		   * lebedev_coords[row_idx + j_n];
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt3Term
(const double exp_lambda, const int coord_idx, const int row_idx,
const int i_m, const int j_m, const int i_n, const int j_n)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * lebedev_coords[row_idx + i_m]
		   * lebedev_coords[row_idx + j_m]
		   * (lebedev_coords[row_idx + i_n]
			  * lebedev_coords[row_idx + i_n]
			  - 
			  lebedev_coords[row_idx + 2]
			  * lebedev_coords[row_idx + 2]);
}



template <typename T, int order, unsigned int vec_dim>
__device__ inline
double
LagrangeMultiplierGPU<T, order, vec_dim>::calcInt4Term
(const double exp_lambda, const int coord_idx, 
const int row_idx, const int i_m, const int j_m)
{
	return exp_lambda * lebedev_weights[coord_idx]
		   * (lebedev_coords[row_idx + i_m]
			  * lebedev_coords[row_idx + i_m]
			  - 
			  lebedev_coords[row_idx + 2]
			  * lebedev_coords[row_idx + 2]);
}

#endif