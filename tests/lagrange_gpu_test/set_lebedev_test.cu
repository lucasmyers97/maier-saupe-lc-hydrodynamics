#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "sphere_lebedev_rule.hpp"
#include <cassert>
#include <algorithm>
#include <string>
#define private public
#include "LagrangeMultiplierGPU.hpp"

namespace{
    constexpr int order{590};
    constexpr int vec_dim{5};
    constexpr int space_dim{3};
}

__global__
void readLebedevGlobal(double *lebedev_coords, double *lebedev_weights, int n_threads,
                       double *out_lebedev_coords, double *out_lebedev_weights)
{
    extern __shared__ LagrangeMultiplierGPU<double, order, vec_dim> lm[];

    int idx = threadIdx.x;
    int block_idx = blockIdx.x;

    int n_lagrange_multipliers = n_threads;
    double *s_lebedev_coords = (double*)&lm[n_lagrange_multipliers];
    double *s_lebedev_weights = (double*)&s_lebedev_coords[space_dim*order];

    lm[idx].readLebedevGlobal(lebedev_coords, lebedev_weights,
                              s_lebedev_coords, s_lebedev_weights,
                              idx, n_threads);

    lm[idx].setLebedevData(s_lebedev_coords, s_lebedev_weights);
    __syncthreads();
    
    int x_coord{};
    int y_coord{};
    int z_coord{};

    if (idx == 0)
    {
        for (int i = 0; i < order; ++i)
        {
            x_coord = i*space_dim;
            y_coord = i*space_dim + 1;
            z_coord = i*space_dim + 2;

            out_lebedev_coords[order*space_dim*block_idx + x_coord] 
                = lm[idx].lebedev_coords[x_coord];
            out_lebedev_coords[order*space_dim*block_idx + y_coord] 
                = lm[idx].lebedev_coords[y_coord];
            out_lebedev_coords[order*space_dim*block_idx + z_coord] 
                = lm[idx].lebedev_coords[z_coord];
            out_lebedev_weights[order*block_idx + i] 
                = lm[idx].lebedev_weights[i];
        }
    }
}

BOOST_AUTO_TEST_CASE(set_lebedev_test)
{
    cudaDeviceProp *prop = new cudaDeviceProp;
    int *device = new int;
    cudaError_t error = cudaGetDevice(device);
    assert(error == 0);
    error = cudaGetDeviceProperties(prop, *device);
    assert(error == 0);

    int max_shared_bytes = 64*1024; // 64 KB
    int space_per_thread 
        = sizeof(LagrangeMultiplierGPU<double, order, vec_dim>);
    int space_for_lebedev = (space_dim + 1)*order*sizeof(double);
    int n_threads = (max_shared_bytes - space_for_lebedev) / space_per_thread;
    n_threads = std::min(n_threads, prop->maxThreadsPerMultiProcessor);
    n_threads = std::min(n_threads, prop->maxThreadsPerBlock);

    int shared_mem_size = n_threads*space_per_thread + space_for_lebedev;

    int n_blocks = prop->multiProcessorCount;
    n_blocks = 24;
    
    cudaFuncSetAttribute(readLebedevGlobal, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         max_shared_bytes);

    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    double *lebedev_coords;
    double *lebedev_weights;
    lebedev_coords = new double[n_blocks*space_dim*order];
    lebedev_weights = new double[n_blocks*order];

    for (int n = 0; n < n_blocks; ++n)
    {
        for (int i = 0; i < order; ++i)
        {
            lebedev_coords[order*space_dim*n + space_dim*i] = x[i];
            lebedev_coords[order*space_dim*n + space_dim*i + 1] = y[i];
            lebedev_coords[order*space_dim*n + space_dim*i + 2] = z[i];
            lebedev_weights[order*n + i] = w[i];
        }
    }

    double *d_lebedev_coords;
    double *d_lebedev_weights;
    cudaMalloc(&d_lebedev_coords, space_dim*order*sizeof(double));
    cudaMalloc(&d_lebedev_weights, order*sizeof(double));

    cudaMemcpy(d_lebedev_coords, lebedev_coords,
               space_dim*order*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lebedev_weights, lebedev_weights,
               order*sizeof(double), cudaMemcpyHostToDevice);

    for (int n = 0; n < n_blocks; ++n)
    {
        for (int i = 0; i < order; ++i)
        {
            lebedev_coords[order*space_dim*n + space_dim*i] = 0;
            lebedev_coords[order*space_dim*n + space_dim*i + 1] = 0;
            lebedev_coords[order*space_dim*n + space_dim*i + 2] = 0;
            lebedev_weights[order*n + i] = 0;
        }
    }
    
    double *d_out_lebedev_coords;
    double *d_out_lebedev_weights;
    cudaMalloc(&d_out_lebedev_coords, n_blocks*space_dim*order*sizeof(double));
    cudaMalloc(&d_out_lebedev_weights, n_blocks*order*sizeof(double));

    readLebedevGlobal <<<n_blocks, n_threads, shared_mem_size>>>
		(d_lebedev_coords, d_lebedev_weights, n_threads,
        d_out_lebedev_coords, d_out_lebedev_weights);

    error = cudaPeekAtLastError();
    BOOST_TEST_REQUIRE(error == 0, 
                       "Error after kernel; error was: " 
                        + std::to_string(error));

    cudaFree(d_lebedev_coords);
    cudaFree(d_lebedev_weights);

    cudaMemcpy(lebedev_coords, d_out_lebedev_coords, 
               n_blocks*space_dim*order*sizeof(double), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(lebedev_weights, d_out_lebedev_weights, 
               n_blocks*order*sizeof(double), cudaMemcpyDeviceToHost);

    error = cudaPeekAtLastError();
    BOOST_TEST_REQUIRE(error == 0, 
                       "Error after copy; error was: " 
                        + std::to_string(error));

    cudaFree(d_out_lebedev_coords);
    cudaFree(d_out_lebedev_weights);
            
    for (int n = 0; n < n_blocks; ++n)
    {
        for (int i = 0; i < order; ++i)
        {
            BOOST_TEST(lebedev_coords[order*space_dim*n + space_dim*i] == x[i]);
            BOOST_TEST(lebedev_coords[order*space_dim*n + space_dim*i + 1] == y[i]);
            BOOST_TEST(lebedev_coords[order*space_dim*n + space_dim*i + 2] == z[i]);
            BOOST_TEST(lebedev_weights[order*n + i] == w[i]);
        }
    }

    std::cout << sizeof(LagrangeMultiplierGPU<double, order, vec_dim>)
              << std::endl;

    delete[] lebedev_coords;
    delete[] lebedev_weights;
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;
}