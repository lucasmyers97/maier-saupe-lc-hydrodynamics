#ifndef PARAMETER_PARSER_HPP
#define PARAMETER_PARSER_HPP

#include <deal.II/base/point.h>

#include <string>
#include <vector>
#include <regex>
#include <sstream>

namespace ParameterParser
{
inline std::vector<std::string> parse_delimited(const std::string &p, 
                                                char delim = ',')
{
    std::istringstream iss(p);
    std::string item;
    std::vector<std::string> list;
    while (std::getline(iss, item, delim))
        list.push_back(item);

    return list;
}



inline void remove_whitespace(std::string &s)
{
    const char* t = " \t\n\r\f\v";
    s.erase(0, s.find_first_not_of(t));
    s.erase(s.find_last_not_of(t) + 1);
}



// template <int dim>
// inline std::vector<dealii::Point<dim>> parse_coordinate_list(std::string &p)
// {
//     // generate pattern -- just all smallest sets of parentheses
//     std::string pattern(R"(\(.*?\))");
//     std::regex re(pattern);
// 
//     // make list of matches
//     auto match_begin = std::sregex_iterator(p.begin(), p.end(), re);
//     auto match_end = std::sregex_iterator();
//     auto dist = std::distance(match_begin, match_end);
// 
//     // parse coordinate matches into vector of coordinates
//     std::vector<dealii::Point<dim>> coords_list(dist);
//     auto coords = coords_list.begin();
//     for (auto match = match_begin; match != match_end; ++match, ++coords)
//     {
//         coords->resize(match->size() - 1);
//         for (std::size_t j = 1; j < match->size(); ++j)
//             (*coords)[j - 1] = std::stod( match->str(j) );
//     }
// }

} // ParameterParser

#endif
