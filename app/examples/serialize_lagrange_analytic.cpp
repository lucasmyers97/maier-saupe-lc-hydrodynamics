#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <fstream>
#include <iostream>
#include <string>

#define private public
#include "Numerics/LagrangeMultiplierAnalytic.hpp"

int main()
{
    const int dim = 3;

    const int order = 590;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const int max_iters = 20;
    const double degenerate_tol = 1e-9;

    LagrangeMultiplierAnalytic<dim> lma(order, alpha, tol,
                                        max_iters, degenerate_tol);

    std::string filename("lagrange_analytic_serialization");
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << lma;
    }
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> lma;
    }

    std::cout << lma.lmr.alpha << " " << lma.lmr.tol << " " << lma.lmr.max_iter
              << " " << lma.lmr.leb.x.size() << " " << lma.degenerate_tol
              << "\n\n";

    return 0;
}
