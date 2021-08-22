#include "LagrangeGPUWrapper.hpp"
#include "LagrangeMultiplierGPU.hpp"
#include "sphere_lebedev_rule.hpp"
#include <cassert>
#include <algorithm>



namespace{
    constexpr int space_dim = 3;
}



template <typename T, unsigned int vec_dim>
__global__
void invertQ(T* d_lebedev_coords, T* d_lebedev_weights, T* Q_in)
{
}



template <typename T, int order, unsigned int vec_dim>
LagrangeGPUWrapper<T, order, vec_dim>::LagrangeGPUWrapper()
{
    initializeLebedevCoords();
}



template <typename T, int order, unsigned int vec_dim>
LagrangeGPUWrapper<T, order, vec_dim>::~LagrangeGPUWrapper()
{
    cudaFree(d_lebedev_coords);
    cudaFree(d_lebedev_weights);
}



template <typename T, int order, unsigned int vec_dim>
void LagrangeGPUWrapper<T, order, vec_dim>::initializeLebedevCoords()
{
    // get initial copy of lebedev data
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];
    ld_by_order(order, x, y, z, w);

    double *lebedev_coords;
    lebedev_coords = new double[space_dim*order];

    // arrange into space_dim x order matrix, row-ordering
    for (int i = 0; i < order; ++i)
    {
        lebedev_coords[space_dim*i] = x[i];
        lebedev_coords[space_dim*i + 1] = y[i];
        lebedev_coords[space_dim*i + 2] = z[i];
    }

    delete[] x;
    delete[] y;
    delete[] z;

    // allocate memory for lebedev data in global gpu
    cudaMalloc(&d_lebedev_coords, space_dim*order*sizeof(T));
    cudaMalloc(&d_lebedev_weights, order*sizeof(T));

    // transfer lebedev data to global gpu
    cudaMemcpy(d_lebedev_coords, lebedev_coords, 
               space_dim*order*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lebedev_weights, w,
               order*sizeof(T), cudaMemcpyHostToDevice);

    delete[] lebedev_coords;
    delete[] w;
}



template <typename T, int order, unsigned int vec_dim>
void LagrangeGPUWrapper<T, order, vec_dim>::getKernelParams()
{
   // get device properties from GPU device
    cudaDeviceProp *prop = new cudaDeviceProp;
    int *device = new int;
    cudaError_t error = cudaGetDevice(device);
    assert(error == 0);
    error = cudaGetDeviceProperties(prop, *device);
    assert(error == 0);

    // figure out device memory allocation based on device properties
    unsigned long max_shared_bytes{};
    if (prop->major >= 7)
    {
        max_shared_bytes = 64*1024; // 64 KB

        // sets max shared memory for kernel to 64kb if architecture allows
        cudaFuncSetAttribute(invertQ<T, vec_dim>, 
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_shared_bytes);
    }
    else
        max_shared_bytes = 48*1024; // 48KB


    // each thread holds one LagrangeMultiplierGPU object
    unsigned long space_per_thread 
        = sizeof(LagrangeMultiplierGPU<T, order, vec_dim>);

    // need space for lebedev coords + weights at the end of shared memory
    unsigned long space_for_lebedev = (space_dim + 1)*order*sizeof(T);

    // n_threads is dictated by min of (1) space needed per thread 
    // (+ lebedev data), (2) max # of threads per multiprocessor,
    /// or (3) max # of threads per block.
    unsigned long n_threads = (max_shared_bytes - space_for_lebedev) 
                              / space_per_thread;
    n_threads = std::min(n_threads, 
                         (unsigned long)prop->maxThreadsPerMultiProcessor);
    n_threads = std::min(n_threads, 
                         (unsigned long)prop->maxThreadsPerBlock);

    // shared memory size based on final n_threads we landed on
    unsigned long shared_mem_size = n_threads*space_per_thread + space_for_lebedev;

    // run one block per shared multiprocessor
    unsigned long n_blocks = prop->multiProcessorCount;

    // save info into kernel_params struct
    kernel_params.n_threads = n_threads;
    kernel_params.n_blocks = n_blocks;
    kernel_params.shared_mem_size = shared_mem_size;
    kernel_params.global_mem_size = prop->totalGlobalMem;

    delete prop;
    delete device;
}

#include "LagrangeGPUWrapper.inst"