#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "sphere_lebedev_rule.hpp"
#include <bitset>
#include <cmath>
#define private public
#include "LagrangeMultiplierGPU.hpp"

namespace utf = boost::unit_test;

namespace{
    constexpr int order{590};
    constexpr int vec_dim{5};
    constexpr int space_dim{3};
    constexpr int n_lambda_vals{32}; // 2^vec_dim
}


__global__
void calcExpLambdaTest
(const double *lebedev_coords, const double *lebedev_weights,
 const double *lambda_vals, double *exp_lambda_vals)
{
    extern __shared__ LagrangeMultiplierGPU<double, order, vec_dim> lm[];
    const int thread_idx = threadIdx.x;
    const int n_threads = blockDim.x;

    // parse shared pointer so very end corresponds to shared lebedev data
    int n_lagrange_multipliers = n_threads;
    double *s_lebedev_coords = (double*)&lm[n_lagrange_multipliers];
    double *s_lebedev_weights = (double*)&s_lebedev_coords[space_dim*order];

    // read lebedev data from global memory into shared memory
    lm[thread_idx].readLebedevGlobal(lebedev_coords, lebedev_weights,
                                     thread_idx, n_threads,
                                     s_lebedev_coords, s_lebedev_weights);

    // make each of the LagrangeMultiplierGPU instances point to shared
    // lebedev data
    lm[thread_idx].setLebedevData(s_lebedev_coords, s_lebedev_weights);

    __syncthreads();

    int row_idx{};
    for (int i = 0; i < n_lambda_vals; ++i)
    {
        for (int j = 0; j < vec_dim; ++j)
            lm[thread_idx].Lambda[j] = lambda_vals[vec_dim*i + j];

        for (int j = 0; j < order; ++j)
        {
            row_idx = space_dim*j;
            exp_lambda_vals[order*i + j] 
                = lm[thread_idx].calcExpLambda(row_idx);
        }
    }
}

double calcExpLambdaVals(double *lambda_vals, double *lebedev_coords)
{
    // construct lambda matrix
    double *lambda = new double[space_dim*space_dim];
    lambda[0] = lambda_vals[0];
    lambda[1] = lambda_vals[1];
    lambda[2] = lambda_vals[2];
    lambda[space_dim] = lambda_vals[1];
    lambda[space_dim + 1] = lambda_vals[3];
    lambda[space_dim + 2] = lambda_vals[4];
    lambda[2*space_dim] = lambda_vals[2];
    lambda[2*space_dim + 1] = lambda_vals[4];
    lambda[2*space_dim + 2] = -(lambda_vals[0] + lambda_vals[3]);

    double lambda_xi_xi{};
    for (int k = 0; k < space_dim; ++k)
        for (int l = 0; l < space_dim; ++l)
            lambda_xi_xi += lambda[space_dim*k + l]
                            * lebedev_coords[k]
                            * lebedev_coords[l];

    delete[] lambda;

    return std::exp(lambda_xi_xi);
}

BOOST_AUTO_TEST_CASE(calc_exp_lambda_test, *utf::tolerance(1e-12))
{
    // get initial copy of lebedev data
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];
    ld_by_order(order, x, y, z, w);

    // make n_blocks copies of lebedev data
    double *lebedev_coords;
    double *lebedev_weights;
    lebedev_coords = new double[space_dim*order];
    lebedev_weights = new double[order];

    for (int i = 0; i < order; ++i)
    {
        lebedev_coords[space_dim*i] = x[i];
        lebedev_coords[space_dim*i + 1] = y[i];
        lebedev_coords[space_dim*i + 2] = z[i];
        lebedev_weights[i] = w[i];
    }

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;

    double *d_lebedev_coords, *d_lebedev_weights;
    cudaMalloc(&d_lebedev_coords, space_dim*order*sizeof(double));
    cudaMalloc(&d_lebedev_weights, order*sizeof(double));

    cudaMemcpy(d_lebedev_coords, lebedev_coords, 
               space_dim*order*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lebedev_weights, lebedev_weights,
               order*sizeof(double), cudaMemcpyHostToDevice);

    double *lambda_vals = new double[n_lambda_vals*vec_dim];
    
    for (int i = 0; i < n_lambda_vals; ++i)
    {
        std::bitset<vec_dim> lambda_bits(i);
        for (int j = 0; j < vec_dim; ++j)
            lambda_vals[vec_dim*i + j] = lambda_bits[j];
    }

    double *d_lambda_vals;
    cudaMalloc(&d_lambda_vals, n_lambda_vals*vec_dim*sizeof(double));
    cudaMemcpy(d_lambda_vals, lambda_vals, 
               n_lambda_vals*vec_dim*sizeof(double), cudaMemcpyHostToDevice);

    double *d_exp_lambda_vals;
    cudaMalloc(&d_exp_lambda_vals, n_lambda_vals*order*sizeof(double));
    
    size_t s_mem_size = sizeof(LagrangeMultiplierGPU<double, order, vec_dim>)
                        + space_dim*order*sizeof(double)
                        + order*sizeof(double);
    calcExpLambdaTest<<<1, 1, s_mem_size>>>
        (d_lebedev_coords, d_lebedev_weights, d_lambda_vals, d_exp_lambda_vals);

    cudaFree(d_lebedev_coords);
    cudaFree(d_lebedev_weights);
    cudaFree(d_lambda_vals);

    double *exp_lambda_vals = new double[n_lambda_vals*order];
    cudaMemcpy(exp_lambda_vals, d_exp_lambda_vals, 
               n_lambda_vals*order*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_exp_lambda_vals);

    for (int i = 0; i < n_lambda_vals; ++i)
    {
        for (int j = 0; j < order; ++j)
        {
            BOOST_TEST(exp_lambda_vals[order*i + j] 
                       == calcExpLambdaVals(&lambda_vals[vec_dim*i], 
                                            &lebedev_coords[space_dim*j]));
        }
    }

    delete[] exp_lambda_vals;
    delete[] lebedev_coords;
    delete[] lebedev_weights;
}