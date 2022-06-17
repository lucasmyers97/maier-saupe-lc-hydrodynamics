#include "Numerics/LagrangeMultiplierReduced.hpp"
#include "Numerics/LagrangeMultiplier.hpp"

#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>

#include <iostream>

#include "Utilities/maier_saupe_constants.hpp"

int main()
{
    namespace msc = maier_saupe_constants;

    const int order = 590;
    const int dim = 3;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const int max_iters = 20;

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    dealii::Tensor<1, 2, double> Q_red;

    Q_vec[0] = -0.1;
    Q_vec[3] = 0.2;
    Q_red[0] = -0.1;
    Q_red[1] = 0.2;

    LagrangeMultiplierReduced lmr(order, alpha, tol, max_iters);
    LagrangeMultiplier<dim> lm(order, alpha, tol, max_iters);

    lmr.invertQ(Q_red);
    lm.invertQ(Q_vec);

    std::cout << (Q_vec[0] - Q_red[0]) * (Q_vec[0] - Q_red[0]) +
        (Q_vec[3] - Q_red[1]) * (Q_vec[3] - Q_red[1]) << "\n\n";

    const int new_order = 974;
    lmr.setOrder(new_order);
    lmr.invertQ(Q_red);

    std::cout << (Q_vec[0] - Q_red[0]) * (Q_vec[0] - Q_red[0]) +
        (Q_vec[3] - Q_red[1]) * (Q_vec[3] - Q_red[1]) << "\n\n";

    return 0;
}
