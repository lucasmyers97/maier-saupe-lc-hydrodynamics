#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "sphere_lebedev_rule/sphere_lebedev_rule.hpp"

#include <deal.II/lac/vector.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <fstream>
#include <cstdio>

#define private public
#include "Numerics/LagrangeMultiplier.hpp"

namespace utf = boost::unit_test;

namespace{
    const int order = 590;
    double alpha = 1.0;
    double tol = 1e-9;
    int max_iters = 15;
    std::string archive_filename = "lagrange_multiplier_archive.dat";
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



BOOST_AUTO_TEST_CASE(test_lagrange_multiplier_inversion)
{
    dealii::Vector<double> Q_vec({0.6,0.0,0.0,-0.3,0.0});
    LagrangeMultiplier<order> lagrange_multiplier(alpha, tol, max_iters);
    lagrange_multiplier.invertQ(Q_vec);
    BOOST_TEST(lagrange_multiplier.inverted);
}



BOOST_AUTO_TEST_CASE(test_lagrange_multiplier_archive)
{
    std::ofstream ofs(archive_filename);

    dealii::Vector<double> Q_vec({0.6, 0.0, 0.0, -0.3, 0.0});
    LagrangeMultiplier<order> lm(alpha, tol, max_iters);
    lm.invertQ(Q_vec);

    {
        boost::archive::text_oarchive oa(ofs);
        oa << lm;
    }

    LagrangeMultiplier<order> new_lm(0, 0, 0);
    {
        std::ifstream ifs(archive_filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> new_lm;
    }

    BOOST_TEST(new_lm.inverted == lm.inverted);
    BOOST_TEST(new_lm.Jac_updated == lm.Jac_updated);
    BOOST_TEST(new_lm.alpha == lm.alpha);
    BOOST_TEST(new_lm.max_iter == lm.max_iter);
    BOOST_TEST(new_lm.Q == lm.Q);
    BOOST_TEST(new_lm.Lambda == lm.Lambda);
    BOOST_TEST(new_lm.Res == lm.Res);
    BOOST_TEST(new_lm.Jac == lm.Jac);
    BOOST_TEST(new_lm.dLambda == lm.dLambda);
    BOOST_TEST(new_lm.Z == lm.Z);
    BOOST_TEST(new_lm.int1 == lm.int1);
    BOOST_TEST(new_lm.int2 == lm.int2);
    BOOST_TEST(new_lm.int3 == lm.int3);
    BOOST_TEST(new_lm.int4 == lm.int4);

    BOOST_TEST(!remove(archive_filename.c_str()));
}
