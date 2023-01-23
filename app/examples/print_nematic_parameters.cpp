/**
 * This file prints parameters of a NematicSystemMPIDriver object to a file
 * in the deal.II PRM format. The file name is the first command-line argument.
 */

#include "SimulationDrivers/NematicSystemMPIDriver.hpp"

#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int av, char *ac[]) {
    try
    {
        if (av - 1 != 2)
            throw std::invalid_argument("Error! Didn't input filename "
                                        "or didn't input style");
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    const int dim = 2;

    dealii::ParameterHandler prm;
    NematicSystemMPIDriver<dim>::declare_parameters(prm);
    NematicSystemMPI<dim>::declare_parameters(prm);

    std::string filename(ac[1]);
    std::string extension = filename.substr(filename.find_last_of('.') + 1);
    std::string style_name(ac[2]);

    dealii::ParameterHandler::OutputStyle style;

    if (style_name == "KeepDeclarationOrder")
        style = dealii::ParameterHandler::OutputStyle::KeepDeclarationOrder;
    else if (style_name == "Short")
        style = dealii::ParameterHandler::OutputStyle::Short;
    else
        throw std::invalid_argument("Wrong style specification");

    if (extension == "prm")
        style = (style | dealii::ParameterHandler::OutputStyle::PRM);
    else if (extension == "tex")
        style = (style | dealii::ParameterHandler::OutputStyle::LaTeX);
    else
        style = (style | dealii::ParameterHandler::OutputStyle::Description);

    prm.print_parameters(filename, style);

    return 0;
}
