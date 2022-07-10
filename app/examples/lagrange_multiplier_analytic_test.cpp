#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <iostream>

#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Numerics/LagrangeMultiplier.hpp"
#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

int main()
{
    const int dim = 3;
    const int order = 974;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const int max_iters = 20;

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    double S = 0.5;
    Q_vec[0] = 0.1;
    Q_vec[1] = -0.03;
    Q_vec[2] = 0;
    Q_vec[3] = 0.05;
    Q_vec[4] = 0;

    LagrangeMultiplierAnalytic<dim> lma(order, alpha, tol, max_iters);
    lma.invertQ(Q_vec);

    dealii::Vector<double> Lambda(msc::vec_dim<dim>);
    dealii::FullMatrix<double> Jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    double Z;

    lma.returnLambda(Lambda);
    lma.returnJac(Jac);
    Z = lma.returnZ();

    std::cout << Lambda << "\n\n";
    Jac.print(std::cout, 15, 5);
    std::cout << "\n" << Z << "\n\n";

    LagrangeMultiplier<dim> lm(order, alpha, tol, max_iters);

    dealii::Vector<double> Lambda_(msc::vec_dim<dim>);
    dealii::FullMatrix<double> Jac_(msc::vec_dim<dim>, msc::vec_dim<dim>);
    dealii::LAPACKFullMatrix<double> lapack_Jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    double Z_;

    lm.invertQ(Q_vec);
    lm.returnLambda(Lambda_);
    lm.returnJac(lapack_Jac);
    Z_ = lm.returnZ();

    lapack_Jac.invert();
    Jac_ = lapack_Jac;

    Lambda_ -= Lambda;
    Jac_.add(-1, Jac);
    Z_ -= Z;
    std::cout << Lambda_ << "\n\n";
    Jac_.print(std::cout, 15, 5);
    std::cout << "\n" << Z_ << "\n\n";

    return 0;
}
