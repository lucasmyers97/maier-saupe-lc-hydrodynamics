#include "Numerics/LagrangeMultiplierEfficient.hpp"
#include "Numerics/LagrangeMultiplier.hpp"

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <vector>
#include <chrono>

#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

int main()
{
    constexpr int order = 974;
    constexpr int dim = 3;
    constexpr int num = 1;

    double alpha = 1.0;
    double tol = 1e-15;
    int max_iter = 30;

    double fudge = 0;

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    Q_vec[0] = -0.12599992319162331;
    // Q_vec[0] = 0.25;
    Q_vec[1] = -0.23851658985824606;
    // Q_vec[1] = 0.05;
    // Q_vec[2] = 0.05;
    Q_vec[2] = fudge;
    // Q_vec[3] = -0.25;
    Q_vec[3] = 0.35103325652486811;
    // Q_vec[4] = 0.05;
    Q_vec[4] = fudge;

    // Q_vec[0] = 0.1;
    // Q_vec[1] = 0.06;
    // Q_vec[2] = 0.08;
    // Q_vec[3] = -0.2;
    // Q_vec[4] = 0.2;

    double Z_eff;
    dealii::Vector<double> Lambda_eff;
    dealii::FullMatrix<double> Jac_eff;

    LagrangeMultiplierEfficient<order, dim> lme(alpha, tol, max_iter);

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < num; ++i)
        lme.invertQ(Q_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = end - start;

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
              << std::endl;

    Z_eff = lme.returnZ();
    lme.returnLambda(Lambda_eff);
    lme.returnJac(Jac_eff);

    // std::cout << Z << std::endl;
    // std::cout << Lambda << std::endl;
    // Jac.print(std::cout, 10, 6);
    // std::cout << std::endl;

    double Z;
    dealii::Vector<double> Lambda;
    dealii::LAPACKFullMatrix<double> Jac_lapack;
    LagrangeMultiplier<order, dim> lm(alpha, tol, max_iter);

    start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < num; ++i)
        lm.invertQ(Q_vec);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
              << std::endl;

    Z = lm.returnZ();
    lm.returnLambda(Lambda);
    lm.returnJac(Jac_lapack);
    Jac_lapack.invert();

    std::cout << (Z_eff - Z) << std::endl;
    Lambda -= Lambda_eff;
    std::cout << Lambda << std::endl;

    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        for (unsigned int j = 0; j < msc::vec_dim<dim>; ++j)
            Jac_eff(i, j) -= Jac_lapack(i, j);

    Jac_eff.print(std::cout, 15, 6);
    std::cout << std::endl;

    return 0;
}
