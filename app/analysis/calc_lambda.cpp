/**
 * This script reads an hdf5 file whose filename is the first command-line
 * argument, from a path in the h5 file given by the second command-line 
 * argument.
 * The h5 file at the path should contain an `n_points`x`msc::vec_dim<dim>`
 * array, where here msc::vec_dim<dim> is 5 (at the current time).
 * Each row should correspond to the degrees of freedom of a Q-tensor
 * evaluated wherever.
 * The program then calculates the Lambda values, and Z values corresponding
 * to those Q-tensors, and writes the calculated values back into the hdf5
 * file.
 */

#include <iostream>
#include <exception>
#include <string>
#include <vector>

#include <highfive/H5Easy.hpp>
#include <deal.II/lac/vector.h>

#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Utilities/maier_saupe_constants.hpp"
namespace msc = maier_saupe_constants;

int main(int ac, char* av[])
{
    try
    {
        if (ac < 2)
        {
            std::cout << "Error! Need to enter .h5 filename with Q data\n";
            return -1;
        }

        std::string filename(av[1]);

        std::string Q_path;
        if (ac < 3)
        {
            Q_path = std::string("Q");
        }
        else if (ac == 3)
        {
            Q_path = std::string(av[2]);
        }
        else
        {
            std::cout << "Error! Entered too many command-line arguments\n";
            return -1;
        }

        using mat = std::vector<std::vector<double>>;
        H5Easy::File file(filename, H5Easy::File::ReadWrite);
        mat Q_vecs = H5Easy::load<mat>(file, Q_path);

        const int dim = 2;
        const unsigned int order = 974;
        const double alpha = 1.0;
        const double tol = 1e-11;
        const int max_iter = 100;

        LagrangeMultiplierAnalytic<dim> lma(order, alpha, tol, max_iter);

        dealii::Vector<double> Q(msc::vec_dim<dim>);
        dealii::Vector<double> Lambda(msc::vec_dim<dim>);
        dealii::FullMatrix<double> dLambda(msc::vec_dim<dim>, 
                                           msc::vec_dim<dim>);
        mat Lambda_vecs(Q_vecs.size(), std::vector<double>(Q_vecs[0].size()));
        std::vector<double> Z(Q_vecs.size());
        std::vector<mat> dLambda_vecs(Q_vecs.size(), 
                                      mat(Q_vecs[0].size(), 
                                          std::vector<double>(Q_vecs[0].size())));
        for (std::size_t i = 0; i < Q_vecs.size(); ++i)
        {
            for (unsigned int j = 0; j < msc::vec_dim<dim>; ++j)
                Q[j] = Q_vecs[i][j];

            lma.invertQ(Q);
            lma.returnLambda(Lambda);
            Z[i] = lma.returnZ();
            lma.returnJac(dLambda);

            for (unsigned int j = 0; j < msc::vec_dim<dim>; ++j)
            {
                Lambda_vecs[i][j] = Lambda[j];

                for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                    dLambda_vecs[i][j][k] = dLambda(j, k);
            }
        }
        
        H5Easy::dump(file, "Lambda", Lambda_vecs, H5Easy::DumpMode::Overwrite);
        H5Easy::dump(file, "Z", Z, H5Easy::DumpMode::Overwrite);
        H5Easy::dump(file, "dLambda", dLambda_vecs, H5Easy::DumpMode::Overwrite);
    }
    catch (const std::exception &exc)
    {
        std::cout << exc.what();
    }

    return 0;
}
