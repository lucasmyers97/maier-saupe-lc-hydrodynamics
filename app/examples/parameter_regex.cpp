#include <iostream>
#include <string>
#include <regex>
#include <vector>

int main()
{
    // read in paramters from std::cin
    std::string parameters;
    std::getline(std::cin, parameters);

    // generate pattern
    std::string pattern(R"(\[((?:\d|\.)+), ((?:\d|\.)+)\])");
    std::regex re(pattern);

    // make list of matches
    auto match_begin = std::sregex_iterator(parameters.begin(),
                                            parameters.end(),
                                            re);
    auto match_end = std::sregex_iterator();
    auto dist = std::distance(match_begin, match_end);

    // parse coordinate matches into vector of coordinates
    std::vector<std::vector<double>> coords_list(dist);
    auto coords = coords_list.begin();
    for (auto match = match_begin; match != match_end; ++match, ++coords)
    {
        coords->resize(match->size() - 1);
        for (std::size_t j = 1; j < match->size(); ++j)
            (*coords)[j - 1] = std::stod( match->str(j) );
    }

    // print out coorindates
    for (const auto &coords : coords_list)
    {
        for (const auto &coord : coords)
            std::cout << coord << " ";
        std::cout << "\n";
    }

    return 0;
}
