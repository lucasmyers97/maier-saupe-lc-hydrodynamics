#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <vector>
#include <cmath>

#include <highfive/H5Easy.hpp>

#include "Numerics/NumericalTools.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Utilities/maier_saupe_constants.hpp"

int main()
{
    namespace msc = maier_saupe_constants;
    constexpr int dim = 2;

    const double begin = {-233.0};
    const double end = {233.0};
    const unsigned int num = {1000};

    std::vector<double> x = NumericalTools::linspace(begin, end, num);

    const double S = 0.6751;
    const double eps = 0.01;
    const double k = 0.0810;
    std::vector<dealii::Vector<double>>
        Q(num, dealii::Vector<double>(msc::vec_dim<dim>));
    for (std::size_t i = 0; i < Q.size(); ++i)
    {
        Q[i][0] = S * (2.0 / 3.0);
        Q[i][1] = S * eps * std::sin(k * x[i]);
        Q[i][3] = S * (-1.0 / 3.0);
    }

    LagrangeMultiplierAnalytic<dim> lma(974, 1.0, 1e-10, 20);
    std::vector<std::vector<double>>
        Lambda_list(num, std::vector<double>(msc::vec_dim<dim>));
    using mat = std::vector<std::vector<double>>;
    std::vector<mat>
        Jac_list(num, mat(msc::vec_dim<dim>,
                          std::vector<double>(msc::vec_dim<dim>)));

    dealii::Vector<double> Lambda(msc::vec_dim<dim>);
    dealii::FullMatrix<double> Jac(msc::vec_dim<dim>, msc::vec_dim<dim>);
    for (std::size_t i = 0; i < Q.size(); ++i)
    {
        lma.invertQ(Q[i]);
        lma.returnLambda(Lambda);
        lma.returnJac(Jac);

        for (std::size_t j = 0; j < msc::vec_dim<dim>; ++j)
        {
            for (std::size_t k = 0; k < msc::vec_dim<dim>; ++k)
                Jac_list[i][j][k] = Jac(j, k);

            Lambda_list[i][j] = Lambda(j);
        }
    }

    H5Easy::File file("periodic_singular_potential.h5",
                      H5Easy::File::Overwrite);
    H5Easy::dump(file, "/Lambda", Lambda_list);
    H5Easy::dump(file, "/Jac", Jac_list);
    H5Easy::dump(file, "/x", x);

    return 0;
}
