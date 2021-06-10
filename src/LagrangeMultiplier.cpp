#include "LagrangeMultiplier.h"
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

LagrangeMultiplier::LagrangeMultiplier(double in_alpha=1)
: alpha(in_alpha)
{
    assert(alpha <= 1);

    if (vec_dim == 5) {
        i(0) = 0;
        i(1) = 0;
        i(2) = 0;
        i(3) = 1;
        i(4) = 1;

        j(0) = 0;
        j(1) = 1;
        j(2) = 2;
        j(3) = 1;
        j(4) = 2;
    }
}

void LagrangeMultiplier::Test()
{
    Matrix<int, mat_dim, mat_dim> m;
    for (int k=0; k<vec_dim; ++k) {
        m(i(k), j(k)) = k;
        if (i(k) != j(k)) {
            m(j(k), i(k)) = k;
        }
    }

    std::cout << m << std::endl;
    std::cout << alpha << std::endl;
}
