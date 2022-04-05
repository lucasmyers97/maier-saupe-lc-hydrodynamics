#include "Numerics/LagrangeMultiplierReduced.hpp"
#include "Numerics/LagrangeMultiplier.hpp"

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>

int main()
{
    const int vec_dim = 5;
    const int dim = 3;
    const int order = 974;

    dealii::Vector<double> Q_vec(vec_dim);
    dealii::Vector<double> Lambda(vec_dim);
    dealii::LAPACKFullMatrix<double> Jac(vec_dim, vec_dim);
    Q_vec[0] = 0.15;
    Q_vec[3] = 0.05;

    LagrangeMultiplier<order, dim> lm(1.0, 1e-10, 10);
    lm.invertQ(Q_vec);
    lm.returnLambda(Lambda);
    lm.returnJac(Jac);

    std::cout << Lambda << std::endl;
    dealii::FullMatrix<double> tmp(vec_dim, vec_dim);
    tmp = Jac;
    tmp.print(std::cout, 10, 6);

    dealii::Tensor<1, 2, double> Q_red;
    dealii::Tensor<1, 2, double> Lambda_red;
    dealii::Tensor<2, 2, double> Jac_red;
    Q_red[0] = 0.15;
    Q_red[1] = 0.05;

    LagrangeMultiplierReduced<order, dim> lmr(1.0, 1e-10, 10);
    lmr.invertQ(Q_red);
    Lambda_red = lmr.returnLambda();
    Jac_red = lmr.returnJac();

    std::cout << Lambda_red << std::endl;
    std::cout << Jac_red << std::endl;

    return 0;
}
