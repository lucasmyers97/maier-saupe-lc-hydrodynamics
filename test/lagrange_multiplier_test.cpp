#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "sphere_lebedev_rule.hpp"
#define private public
#include "LagrangeMultiplier.hpp"

namespace utf = boost::unit_test;

namespace{
    const int order = 590;
    double alpha = 1.0;
    double tol = 1e-9;
    int max_iters = 15;
}

BOOST_AUTO_TEST_CASE(test_lebedev_weights_coords, *utf::tolerance(1e-12))
{
    double *x, *y, *z, *w;
    x = new double[order];
    y = new double[order];
    z = new double[order];
    w = new double[order];

    ld_by_order(order, x, y, z, w);

    LagrangeMultiplier<order> lm(alpha, tol, max_iters);

    for (int quad_idx = 0; quad_idx < order; ++quad_idx)
    {
        BOOST_TEST(lm.lebedev_coords[quad_idx][0] == x[quad_idx]);
        BOOST_TEST(lm.lebedev_coords[quad_idx][1] == y[quad_idx]);
        BOOST_TEST(lm.lebedev_coords[quad_idx][2] == z[quad_idx]);
        BOOST_TEST(lm.lebedev_weights[quad_idx] == w[quad_idx]);
    }

    delete[] x;
    delete[] y;
    delete[] z;
    delete[] w;
}