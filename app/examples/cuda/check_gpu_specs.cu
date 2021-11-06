#include <iostream>
#include <cassert>

int main()
{
    cudaDeviceProp *prop = new cudaDeviceProp;
    int *device = new int;
    cudaError_t error = cudaGetDevice(device);
    assert(error == 0);
    error = cudaGetDeviceProperties(prop, *device);
    assert(error == 0);

    std::cout << "Total global memory: " 
              << prop->totalGlobalMem << std::endl;
    std::cout << "Maximum threads per multiprocessor: "
              << prop->maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum threads per block: "
              << prop->maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per multiprocessor: "
              << prop->sharedMemPerMultiprocessor << std::endl;
    std::cout << "Shared memory per block: "
              << prop->sharedMemPerBlock << std::endl;
    std::cout << "Number of multiprocessors: "
              << prop->multiProcessorCount << std::endl;

    std::cout << "Max dynamic shared memory size: "
              << cudaFuncAttributeMaxDynamicSharedMemorySize << std::endl;
    std::cout << "Preferred shared memory carveout: "
              << cudaFuncAttributePreferredSharedMemoryCarveout << std::endl;

    return 0;
}