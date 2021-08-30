#ifndef LAGRANGE_GPU_WRAPPER_HPP
#define LAGRANGE_GPU_WRAPPER_HPP

#include "LagrangeMultiplierGPU.hpp"
#include <cstddef>


namespace{
    std::size_t space_dim = 3;
}



template <typename T, int order, unsigned int vec_dim>
class LagrangeGPUWrapper
{
public:
    struct KernelParams {
        unsigned long n_blocks;
        unsigned long n_threads;
        unsigned long shared_mem_size;
        unsigned long global_mem_size;
    } kernel_params;

    __host__ LagrangeGPUWrapper(const std::size_t n_Q_vals_in, 
                                double tol_in, int max_iters_in);
    __host__ ~LagrangeGPUWrapper();
    __host__ void getKernelParams();

    __host__ void sendQToDevice();
    __host__ void readQFromDevice();
    __host__ void runKernel();

    __host__ void invertQValues();

    __host__ inline double& operator()(std::size_t component_idx,
                                       std::size_t array_idx)
    {
        return Q_array[vec_dim*array_idx + component_idx];
    }

private:
    __host__ void initializeLebedevCoords();

    T *d_lebedev_coords;
    T *d_lebedev_weights;
    std::size_t n_Q_vals;
    double tol;
    int max_iters;
    T *Q_array;
    T* d_Q_array;
};



template <typename T, int order, unsigned int vec_dim>
__device__ inline
void initializeSharedMemory
(const double *g_lebedev_coords, const double *g_lebedev_weights,
const std::size_t n_threads, const size_t idx,
LagrangeMultiplierGPU<T, order, vec_dim> *lm)
{
    int space_dim = 3;

    // parse shared pointer so very end corresponds to shared lebedev data
    int n_lagrange_multipliers = n_threads;
    double *s_lebedev_coords = (double*)&lm[n_lagrange_multipliers];
    double *s_lebedev_weights = (double*)&s_lebedev_coords[space_dim*order];

    // read lebedev data from global memory into shared memory
    lm[idx].readLebedevGlobal(g_lebedev_coords, g_lebedev_weights,
                              idx, n_threads,
                              s_lebedev_coords, s_lebedev_weights);

    // make each of the LagrangeMultiplierGPU instances point to shared
    // lebedev data
    lm[idx].setLebedevData(s_lebedev_coords, s_lebedev_weights);
    __syncthreads();
}



template <class T>
__device__ T* shared_memory_proxy()
{
    // just give the shared memory some name, and some type
    extern __shared__ unsigned char memory[];

    // reinterpret the shared memory as class T*
    return reinterpret_cast<T*>(memory);
}



template <typename T, int order, unsigned int vec_dim>
__global__
void invertQKernel
(const double *d_lebedev_coords, const double *d_lebedev_weights,
const double tol, const int max_iters, const std::size_t n_Q_vecs, 
double *d_Q_array)
{
    auto lm = shared_memory_proxy<LagrangeMultiplierGPU<T, order, vec_dim>>();

    size_t thread_idx = threadIdx.x;
    size_t n_threads = blockDim.x;
    size_t block_idx = blockIdx.x;
    size_t global_idx = blockDim.x * block_idx + thread_idx;
    size_t global_stride = blockDim.x * gridDim.x;

    initializeSharedMemory<T, order, vec_dim>(d_lebedev_coords, d_lebedev_weights,
                                              n_threads, thread_idx, lm);

    lm[thread_idx].setParams(tol, max_iters);
    for (int i = global_idx; i < n_Q_vecs; i += global_stride)
        lm[thread_idx].calcLambda(&d_Q_array[i*vec_dim]);
}

#endif