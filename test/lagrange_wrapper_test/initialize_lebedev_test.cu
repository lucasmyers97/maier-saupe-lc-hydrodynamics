#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "sphere_lebedev_rule.hpp"
#define private public
#include "LagrangeGPUWrapper.hpp"

namespace utf = boost::unit_test;



namespace{
    constexpr int order = 590;
    using T = double;
    constexpr int vec_dim = 5;
}



BOOST_AUTO_TEST_CASE(initialize_lebedev_test, *utf::tolerance(1e-12))
{
    std::size_t n_pts = 1;
    double tol = 1e-9;
    int max_iters = 12;
    LagrangeGPUWrapper<T, order, vec_dim> lmw(n_pts, tol, max_iters);
    T *lebedev_coords = new double[space_dim*order];
    T *lebedev_weights = new double[order];

    cudaMemcpy(lebedev_coords, lmw.d_lebedev_coords,
               space_dim*order*sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(lebedev_weights, lmw.d_lebedev_weights,
               order*sizeof(T), cudaMemcpyDeviceToHost);

    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    for (int i = 0; i < order; ++i)
    {
        BOOST_TEST(lebedev_coords[space_dim*i] == x[i]);
        BOOST_TEST(lebedev_coords[space_dim*i + 1] == y[i]);
        BOOST_TEST(lebedev_coords[space_dim*i + 2] == z[i]);
        BOOST_TEST(lebedev_weights[i] == w[i]);
    }

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;

    delete[] lebedev_coords;
    delete[] lebedev_weights;
}