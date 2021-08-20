#ifndef LAGRANGE_GPU_WRAPPER_HPP
#define LAGRANGE_GPU_WRAPPER_HPP

#include "LagrangeMultiplierGPU.hpp"



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

    __host__ LagrangeGPUWrapper();
    __host__ void getKernelParams();
    __host__ void transferQValues();
    __host__ void runKernel();
    __host__ void readQValues();
    __host__ void invertQValues();
    __device__ void initializeSharedMemory
                    (LagrangeMultiplierGPU<T, order, vec_dim> *lm,
                     double *g_lebedev_coords,
                     double *g_lebedev_weights);

private:
    __host__ void initializeLebedevCoords();

    T *d_lebedev_coords;
    T *d_lebedev_weights;
};

#endif