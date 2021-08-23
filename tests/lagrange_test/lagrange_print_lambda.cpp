#include <boost/test/tools/interface.hpp>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <deal.II/lac/vector.h>
#define private public
#include "LagrangeMultiplier.hpp"

namespace {
    constexpr int order{590};
    constexpr double alpha{1.0};
    constexpr double tol{1e-9};
    constexpr int max_iter{15};

    constexpr int vec_dim{5};
}

BOOST_AUTO_TEST_CASE(lagrange_print_lambda)
{
    LagrangeMultiplier<order> lm(alpha, tol, max_iter);
    dealii::Vector<double> Q(vec_dim);

    Q(0) = 0.6;
    Q(1) = 0;
    Q(2) = 0;
    Q(3) = -0.3;
    Q(4) = 0;

    lm.setQ(Q);
    lm.updateRes();

    int iter{0};
    while (lm.Res.l2_norm() > tol && iter < max_iter)
    {
        lm.updateJac();
        lm.updateVariation();
        lm.dLambda *= alpha;
        lm.Lambda -= lm.dLambda;
        lm.updateRes();

        // std::cout << lm.Res.l2_norm() << std::endl;
    }

    BOOST_TEST(lm.Res.l2_norm() < tol);
}