#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#define private public
#include "LagrangeGPUWrapper.hpp"



namespace{
    constexpr int order = 590;
    using T = double;
    constexpr int vec_dim = 5;
}



BOOST_AUTO_TEST_CASE(get_kernel_params_test)
{
    std::size_t n_pts{1};
    double tol = 1e-9;
    int max_iters = 12;
    LagrangeGPUWrapper<T, order, vec_dim> lmw(n_pts, tol, max_iters);
    lmw.getKernelParams();

   // get device properties from GPU device
    cudaDeviceProp *prop = new cudaDeviceProp;
    int *device = new int;
    cudaError_t error = cudaGetDevice(device);
    assert(error == 0);
    error = cudaGetDeviceProperties(prop, *device);
    assert(error == 0);

    BOOST_TEST(lmw.kernel_params.global_mem_size == prop->totalGlobalMem);
    BOOST_TEST(lmw.kernel_params.n_blocks == prop->multiProcessorCount);

    delete prop;
    delete device;
}