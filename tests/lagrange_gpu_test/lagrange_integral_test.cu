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
void testLagrangeIntegrals(const double *lebedev_coords, 
                           const double *lebedev_weights,
                           double *int1, double *int2,
                           double *int3, double *int4)
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

    // (i, j) components in Q-matrix for each degree of freedom
    int i[vec_dim] = {0, 0, 0, 1, 1};
	int j[vec_dim] = {0, 1, 2, 1, 2};

    int row_idx{};
    for (int coord_idx = 0; coord_idx < order; ++coord_idx)
    {
        for (int m = 0; m < vec_dim; ++m)
        {
            row_idx = space_dim*coord_idx;
            int1[m] += lm[thread_idx].calcInt1Term(1, coord_idx, row_idx, 
                                                   i[m], j[m]);
            int4[m] += lm[thread_idx].calcInt4Term(1, coord_idx, row_idx, 
                                                   i[m], j[m]);

            for (int n = 0; n < vec_dim; ++n)
            {
                int2[vec_dim*m + n] 
                    += lm[thread_idx].calcInt2Term(1, coord_idx, row_idx, 
                                                   i[m], j[m], i[n], j[n]);
                int3[vec_dim*m + n] 
                    += lm[thread_idx].calcInt3Term(1, coord_idx, row_idx, 
                                                   i[m], j[m], i[n], j[n]);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(lagrange_integral_test, *utf::tolerance(1e-12))
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

    double *d_int1, *d_int2, *d_int3, *d_int4;
    cudaMalloc(&d_int1, vec_dim*sizeof(double));
    cudaMalloc(&d_int2, vec_dim*vec_dim*sizeof(double));
    cudaMalloc(&d_int3, vec_dim*vec_dim*sizeof(double));
    cudaMalloc(&d_int4, vec_dim*sizeof(double));

    size_t s_mem_size = sizeof(LagrangeMultiplierGPU<double, order, vec_dim>)
                        + space_dim*order*sizeof(double)
                        + order*sizeof(double);
    testLagrangeIntegrals <<<1, 1, s_mem_size>>>
        (d_lebedev_coords, d_lebedev_weights, d_int1, d_int2, d_int3, d_int4);

    cudaFree(d_lebedev_coords);
    cudaFree(d_lebedev_weights);

    double *int1, *int2, *int3, *int4;
    int1 = new double[vec_dim*sizeof(double)];
    int2 = new double[vec_dim*vec_dim*sizeof(double)];
    int3 = new double[vec_dim*vec_dim*sizeof(double)];
    int4 = new double[vec_dim*sizeof(double)];

    cudaMemcpy(int1, d_int1, vec_dim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(int2, d_int2, 
               vec_dim*vec_dim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(int3, d_int3, 
               vec_dim*vec_dim*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(int4, d_int4, vec_dim*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_int1);
    cudaFree(d_int2);
    cudaFree(d_int3);
    cudaFree(d_int4);

    for (int m = 0; m < vec_dim; ++m)
    {
        if (m == 0 || m == 3)
            BOOST_TEST(int1[m] == 1.0/3.0);
        else
            BOOST_TEST(int1[m] == 0.0);

        BOOST_TEST(int4[m] == 0.0);

        for (int n = 0; n < vec_dim; ++n)
        {
            if (m == n)
            {
                if (n == 0 || n == 3)
                    BOOST_TEST(int3[vec_dim*m + n] == 2.0/15.0);
                else
                    BOOST_TEST(int2[vec_dim*m + n] == 1.0/15.0);;
            }
            else 
            {
                if (n == 0 || n == 3)
                    BOOST_TEST(int3[vec_dim*m + n] == 0.0);
                else
                    BOOST_TEST(int2[vec_dim*m + n] == 0.0);
            }
        }
    }

    delete[] int1;
    delete[] int2;
    delete[] int3;
    delete[] int4;
}