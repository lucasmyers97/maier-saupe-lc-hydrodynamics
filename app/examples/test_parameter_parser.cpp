/**
 * Tests functions in ParameterParser utility library
 */

#include "Utilities/ParameterParser.hpp"

#include <string>
#include <iostream>

int main()
{
    std::string p("3.2, 0.1, my mom, fifty, 3");
    auto list = ParameterParser::parse_delimited(p);

    for (const auto &word : list)
        std::cout << word << "\n";

    std::cout << "\n";

    for (auto &word : list)
    {
        ParameterParser::remove_whitespace(word);
        std::cout << word << "\n";
    }

    constexpr int dim = 2;
    std::string q("[0.3, 2.5] [72, 38]");
    std::vector<dealii::Point<dim>> coords
        = ParameterParser::parse_coordinate_list<dim>(q);

    for (const auto &coord : coords)
        std::cout << coord << std::endl;

    return 0;
}
