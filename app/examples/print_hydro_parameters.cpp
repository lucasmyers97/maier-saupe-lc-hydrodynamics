/**
 * This file prints parameters of a HydroSystemMPI object to a file
 * in the deal.II PRM format. The file name is the first command-line argument.
 */

#include "LiquidCrystalSystems/HydroSystemMPI.hpp"

#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int av, char *ac[]) {
    try
    {
        if (av - 1 != 1)
            throw std::invalid_argument("Error! Didn't input filename");
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    const int dim = 2;

    dealii::ParameterHandler prm;
    HydroSystemMPI<dim>::declare_parameters(prm);

    std::string filename(ac[1]);
    prm.print_parameters(filename);

    return 0;
}
