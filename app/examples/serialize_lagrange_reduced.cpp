#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <fstream>
#include <string>
#include <iostream>

#define private public
#include "Numerics/LagrangeMultiplierReduced.hpp"

int main()
{
    const std::string filename("lagrange_reduced_serial");

    const int order = 590;
    const double alpha = 1.0;
    const double tol = 1e-10;
    const int max_iters = 20;

    LagrangeMultiplierReduced lmr(order, alpha, tol, max_iters);

    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << lmr;
    }
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> lmr;
    }

    std::cout << lmr.alpha << " " << lmr.tol << " " << lmr.max_iter << " "
              << lmr.leb.x.size() << "\n\n";

    return 0;
}
