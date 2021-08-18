#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "sphere_lebedev_rule.hpp"
#define private public
#include "LagrangeMultiplierGPU.hpp"

namespace utf = boost::unit_test;

namespace{
    constexpr int order{590};
    constexpr int vec_dim{5};
    constexpr int space_dim{3};
}

__global__
void calcResJacTest
(const double *lebedev_coords, const double *lebedev_weights,
 double *Res, double *Jac)
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

    double *Q = new double[vec_dim];
    Q[0] = 2.0/3.0;
    Q[1] = 0;
    Q[2] = 0;
    Q[3] = -1.0/3.0;
    Q[4] = 0;

    lm[thread_idx].initializeInversion(Q);
    for (int i = 0; i < vec_dim; ++i)
        lm[thread_idx].Lambda[i] = 0;

    lm[thread_idx].calcResJac();

    for (int i = 0; i < vec_dim; ++i)
    {
        Res[i] = lm[thread_idx].Res[i];

        for (int j = 0; j < vec_dim; ++j)
        {
            Jac[vec_dim*i + j] = lm[thread_idx].Jac(i, j);
        }
    }

    delete[] Q;
}

BOOST_AUTO_TEST_CASE(calc_res_jac_test, *utf::tolerance(1e-12))
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

    delete[] lebedev_coords;
    delete[] lebedev_weights;

    double *d_Res, *d_Jac;
    cudaMalloc(&d_Res, vec_dim*sizeof(double));
    cudaMalloc(&d_Jac, vec_dim*vec_dim*sizeof(double));

    size_t s_mem_size = sizeof(LagrangeMultiplierGPU<double, order, vec_dim>)
                        + space_dim*order*sizeof(double)
                        + order*sizeof(double);
    calcResJacTest<<<1, 1, s_mem_size>>>
        (d_lebedev_coords, d_lebedev_weights, d_Res, d_Jac);

    double *Res = new double[vec_dim];
    double *Jac = new double[vec_dim*vec_dim];

    cudaMemcpy(Res, d_Res, vec_dim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Jac, d_Jac, 
               vec_dim*vec_dim*sizeof(double), cudaMemcpyDeviceToHost);

    double *Q = new double[vec_dim];
    Q[0] = 2.0/3.0;
    Q[1] = 0;
    Q[2] = 0;
    Q[3] = -1.0/3.0;
    Q[4] = 0;
    for (int m = 0; m < vec_dim; ++m)
    {
        BOOST_TEST(Res[m] == -Q[m]);

        for (int n = 0; n < vec_dim; ++n)
        {
            if (m == n)
                BOOST_TEST(Jac[vec_dim*m + n] == 2.0/15.0);
            else
                BOOST_TEST(Jac[vec_dim*m + n] == 0);
        }
    }

    delete[] Q;
}