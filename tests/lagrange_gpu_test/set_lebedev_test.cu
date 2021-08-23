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
    /**
    * Reads Lebedev coordinates and weights in from global memory to shared
    * memory for any given block of threads. For testing, it outputs each set
    * of Lebedev coordinates and weights (that is, corresponding to each block)
    * in its place in a new global array. This will be copied to the host and
    * checked to make sure every block's shared memory has a copy of the
    * Lebedev data.
    */
    extern __shared__ LagrangeMultiplierGPU<double, order, vec_dim> lm[];

    int idx = threadIdx.x;
    int block_idx = blockIdx.x;

    // parse shared pointer so very end corresponds to shared lebedev data
    int n_lagrange_multipliers = n_threads;
    double *s_lebedev_coords = (double*)&lm[n_lagrange_multipliers];
    double *s_lebedev_weights = (double*)&s_lebedev_coords[space_dim*order];

    // read lebedev data from global memory into shared memory
    lm[idx].readLebedevGlobal(lebedev_coords, lebedev_weights,
                              idx, n_threads,
                              s_lebedev_coords, s_lebedev_weights);

    // make each of the LagrangeMultiplierGPU instances point to shared
    // lebedev data
    lm[idx].setLebedevData(s_lebedev_coords, s_lebedev_weights);
    __syncthreads();
    
    int x_coord{};
    int y_coord{};
    int z_coord{};

    // make 0th thread in each block write shared lebedev data to global
    // output array for lebedev data -- each block gets different place in
    // global output array
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
    // get device properties from GPU device
    cudaDeviceProp *prop = new cudaDeviceProp;
    int *device = new int;
    cudaError_t error = cudaGetDevice(device);
    assert(error == 0);
    error = cudaGetDeviceProperties(prop, *device);
    assert(error == 0);

    delete device;

    // figure out device memory allocation based on device properties
    int max_shared_bytes = 64*1024; // 64 KB

    // each thread holds one LagrangeMultiplierGPU object
    int space_per_thread 
        = sizeof(LagrangeMultiplierGPU<double, order, vec_dim>);

    // need space for lebedev coords + weights at the end of shared memory
    int space_for_lebedev = (space_dim + 1)*order*sizeof(double);

    // n_thread is dictated by min of (1) space needed per thread 
    // (+ lebedev data), (2) max # of threads per multiprocessor,
    /// or (3) max # of threads per block.
    int n_threads = (max_shared_bytes - space_for_lebedev) / space_per_thread;
    n_threads = std::min(n_threads, prop->maxThreadsPerMultiProcessor);
    n_threads = std::min(n_threads, prop->maxThreadsPerBlock);

    int shared_mem_size = n_threads*space_per_thread + space_for_lebedev;
    int n_blocks = prop->multiProcessorCount;

    delete prop;
    
    // sets max shared memory to 64kb
    cudaFuncSetAttribute(readLebedevGlobal, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 
                         max_shared_bytes);

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

    // allocate device input lebedev data for kernel
    double *d_lebedev_coords;
    double *d_lebedev_weights;
    cudaMalloc(&d_lebedev_coords, space_dim*order*sizeof(double));
    cudaMalloc(&d_lebedev_weights, order*sizeof(double));

    cudaMemcpy(d_lebedev_coords, lebedev_coords,
               space_dim*order*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lebedev_weights, lebedev_weights,
               order*sizeof(double), cudaMemcpyHostToDevice);

    // zero out host lebedev data so we can use later for check
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
    
    // allocated device output lebedev data for kernel
    double *d_out_lebedev_coords;
    double *d_out_lebedev_weights;
    cudaMalloc(&d_out_lebedev_coords, n_blocks*space_dim*order*sizeof(double));
    cudaMalloc(&d_out_lebedev_weights, n_blocks*order*sizeof(double));

    readLebedevGlobal <<<n_blocks, n_threads, shared_mem_size>>>
		(d_lebedev_coords, d_lebedev_weights, n_threads,
        d_out_lebedev_coords, d_out_lebedev_weights);

    // make sure there were no cuda errors for kernel
    error = cudaPeekAtLastError();
    BOOST_TEST_REQUIRE(error == 0, 
                       "Error after kernel; error was: " 
                        + std::to_string(error));

    cudaFree(d_lebedev_coords);
    cudaFree(d_lebedev_weights);

    // copy output device data to host and check for errors
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
            
    // check host copies of lebedev data to make sure all blocks copied data
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

    // free up remaining host pointers
    delete[] lebedev_coords;
    delete[] lebedev_weights;
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;
}