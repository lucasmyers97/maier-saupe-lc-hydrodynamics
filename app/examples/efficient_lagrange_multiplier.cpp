#include "Numerics/LagrangeMultiplierEfficient.hpp"
#include "Numerics/LagrangeMultiplier.hpp"

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <vector>
#include <chrono>
#include <limits>

#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

int main()
{
    constexpr int order = 974;
    constexpr int dim = 3;
    constexpr int num = 100;

    double alpha = 1.0;
    double tol = 1e-14;
    int max_iter = 30;

    double fudge = 0;

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    // Q_vec[0] = -0.12599992319162331;
    // // Q_vec[0] = 0.25;
    // Q_vec[1] = -0.23851658985824606;
    // // Q_vec[1] = 0.05;
    // // Q_vec[2] = 0.05;
    // Q_vec[2] = fudge;
    // // Q_vec[3] = -0.25;
    // Q_vec[3] = 0.35103325652486811;
    // // Q_vec[4] = 0.05;
    // Q_vec[4] = fudge;

    // Q_vec[0] = -0.219537;
    // Q_vec[1] = -0.0664242;
    // Q_vec[2] = 0;
    // Q_vec[3] = 0.444571;
    // Q_vec[4] = 0;

    Q_vec[0] = -0.05;
    // Q_vec[1] = -0.0488917;
    Q_vec[1] = 0;
    Q_vec[2] = 0;
    Q_vec[3] = 0.1;
    Q_vec[4] = 0;

    double Z_eff;
    dealii::Vector<double> Lambda_eff;
    dealii::FullMatrix<double> Jac_eff;
    LagrangeMultiplierEfficient<dim> lme(order, alpha, tol, max_iter);

    double Z;
    dealii::Vector<double> Lambda;
    dealii::LAPACKFullMatrix<double> Jac_lapack;
    LagrangeMultiplier<dim> lm(order, alpha, tol, max_iter);

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < num; ++i)
        lme.invertQ(Q_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = end - start;

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
              << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < num; ++i)
        lm.invertQ(Q_vec);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
              << std::endl;

    Z_eff = lme.returnZ();
    lme.returnLambda(Lambda_eff);
    lme.returnJac(Jac_eff);

    Z = lm.returnZ();
    lm.returnLambda(Lambda);
    lm.returnJac(Jac_lapack);
    Jac_lapack.invert();

    std::cout << (Z_eff - Z) << std::endl;
    Lambda -= Lambda_eff;
    std::cout << Lambda.l2_norm() << std::endl;
    std::cout << Lambda << std::endl;

    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        for (unsigned int j = 0; j < msc::vec_dim<dim>; ++j)
            Jac_eff(i, j) -= Jac_lapack(i, j);

    Jac_eff.print(std::cout, 15, 6);
    std::cout << std::endl;

    // This is for checking on different Q-configurations, not timing

    // for (unsigned int i = 0; i < num; ++i)
    // {
    //     double S = 0.6751;
    //     double phi = (M_PI * static_cast<double>(i))
    //         / static_cast<double>(num - 1);

    //     Q_vec[0] = 0.5 * S * (1.0 / 3.0 + std::cos(2 * phi));
    //     Q_vec[1] = 0.5 * S * std::sin(2 * phi);
    //     Q_vec[2] = 0;
    //     Q_vec[3] = 0.5 * S * (1.0 / 3.0 - std::cos(2 * phi));
    //     Q_vec[4] = 0;

    //     // std::cout << Q_vec << std::endl;

    //     lme.invertQ(Q_vec);
    //     lm.invertQ(Q_vec);

    //     Z_eff = lme.returnZ();
    //     lme.returnLambda(Lambda_eff);
    //     lme.returnJac(Jac_eff);

    //     Z = lm.returnZ();
    //     lm.returnLambda(Lambda);
    //     lm.returnJac(Jac_lapack);
    //     Jac_lapack.invert();

    //     // std::cout << (Z_eff - Z) << std::endl;
    //     Lambda -= Lambda_eff;
    //     std::cout << Lambda.l2_norm() << std::endl;
    //     std::cout << Lambda_eff << std::endl;
    //     std::cout << Lambda << std::endl;

    //     for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
    //         for (unsigned int j = 0; j < msc::vec_dim<dim>; ++j)
    //             Jac_eff(i, j) -= Jac_lapack(i, j);

    //     Jac_eff.print(std::cout, 15, 6);
    //     std::cout << std::endl;

    // }

    return 0;
}
