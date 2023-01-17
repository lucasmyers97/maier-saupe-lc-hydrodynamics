/**
 * Tests functions in ParameterParser utility library
 */

#include "Utilities/ParameterParser.hpp"

#include <string>
#include <iostream>

int main()
{
    std::string p("3.2, 0.1, my mom, fifty, 3");
    const auto list = ParameterParser::parse_delimited(p);

    for (const auto &word : list)
        std::cout << word << "\n";

    return 0;
}
