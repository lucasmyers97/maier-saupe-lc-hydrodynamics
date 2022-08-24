/**
 * This program just tests out the serialization of all of the derived
 * BoundaryValues classes.
 * Mostly this is just to test if I will immediately run into problems trying
 * to serialize the LiquidCrystalSystems which have BoundaryValue methods.
 */

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <fstream>
#include <iostream>
#include <string>

#define private public

#include "BoundaryValues/DefectConfiguration.hpp"

int main()
{
    const int dim = 2;
    const double S = 0.5;
    const DefectCharge charge = DefectCharge::plus_half;
    const std::string filename = "boundary_values_filename";

    {
        std::ofstream ofs(filename);
        DefectConfiguration<dim> defect_configuration(S, charge);
        boost::archive::text_oarchive oa(ofs);
        oa << defect_configuration;
    }

    {
        std::ifstream ifs(filename);
        DefectConfiguration<dim> defect_configuration;
        boost::archive::text_iarchive ia(ifs);
        ia >> defect_configuration;

        std::cout << (defect_configuration.S0 == S) << "\n";
        std::cout << (defect_configuration.charge == charge) << "\n";
    }


    return 0;
}
