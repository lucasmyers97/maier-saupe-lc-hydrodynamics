#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "LagrangeGPUWrapper.hpp"
#include <cmath>



BOOST_AUTO_TEST_CASE(invert_Q_test)
{
    constexpr int n_Q_pts = 10;
    constexpr unsigned int vec_dim = 5;
    constexpr int order = 590;
    double tol = 1e-9;
    int max_iters = 12;

    LagrangeGPUWrapper<double, order, vec_dim> lagrange_wrapper(n_Q_pts, tol, max_iters);
    
    for(int i = 0; i < n_Q_pts; ++i)
    {
        double theta = double(i) / double(n_Q_pts) * M_PI;
        double S = 0.9;

        lagrange_wrapper(0, i) = S * (cos(theta)*cos(theta) - 1.0/3.0);
        lagrange_wrapper(1, i) = S * cos(theta)*sin(theta);
        lagrange_wrapper(2, i) = S * 0;
        lagrange_wrapper(3, i) = S * (sin(theta)*sin(theta) - 1.0/3.0);
        lagrange_wrapper(4, i) = S * 0;
    }

    lagrange_wrapper.invertQValues();
    cudaError_t error = cudaPeekAtLastError();

    BOOST_TEST(error == 0);
}