#ifndef PARAMETER_PARSER_HPP
#define PARAMETER_PARSER_HPP

#include <deal.II/base/point.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <regex>
#include <sstream>
#include <iostream>

namespace ParameterParser
{
inline void remove_whitespace(std::string &s)
{
    const char* t = " \t\n\r\f\v";
    s.erase(0, s.find_first_not_of(t));
    s.erase(s.find_last_not_of(t) + 1);
}



inline std::vector<std::string> parse_delimited(const std::string &p, 
                                                char delim = ',')
{
    std::istringstream iss(p);
    std::string item;
    std::vector<std::string> list;
    while (std::getline(iss, item, delim))
    {
        remove_whitespace(item);
        list.push_back(item);
    }

    return list;
}



template <int dim>
inline std::vector<std::vector<double>> parse_coordinate_list(const std::string &p)
{
    // generate pattern -- just all smallest sets of parentheses
    std::string pattern(R"(\[(.*?)\])");
    std::regex re(pattern);

    // make list of matches
    auto match_begin = std::sregex_iterator(p.begin(), p.end(), re);
    auto match_end = std::sregex_iterator();
    auto dist = std::distance(match_begin, match_end);

    // parse coordinate matches into vector of coordinates
    std::vector<std::vector<double>> coords_list(dist, 
                                                 std::vector<double>(dim));
    auto coords = coords_list.begin();
    for (auto match = match_begin; match != match_end; ++match, ++coords)
    {
        std::vector<std::string> coords_string 
            = parse_delimited(match->str(1));
        if (coords_string.size() != dim)
            throw std::runtime_error(std::string("coordinate list is wrong "
                                                 "length in parse_coordinate_list"));

        for (std::size_t i = 0; i < coords_string.size(); ++i)
        {
            (*coords)[i] = std::stod(coords_string[i]);
        }
    }

    return coords_list;
}



inline std::vector<double> parse_number_list(const std::string &p, 
                                             char delim = ',')
{
    std::istringstream iss(p);
    std::string item;
    std::vector<double> list;
    while (std::getline(iss, item, delim))
    {
        remove_whitespace(item);
        list.push_back( std::stod(item) );
    }

    return list;
}



template <int dim>
inline std::vector<dealii::Point<dim>>
vector_to_dealii_point(const std::vector<std::vector<double>> &vec_list)
{
    std::vector<dealii::Point<dim>> points;
    points.reserve(vec_list.size());
    for (const auto &vec : vec_list)
        points.push_back( dim == 2 ?
                dealii::Point<dim>(vec[0], vec[1]) :
                dealii::Point<dim>(vec[0], vec[1], vec[2])
                );

    return points;
}



template <int dim>
inline std::vector<std::vector<double>>
dealii_point_to_vector(const std::vector<dealii::Point<dim>> &point_list)
{
    std::vector<std::vector<double>> vec_list(point_list.size(),
                                              std::vector<double>(dim));

    for (std::size_t i = 0; i < point_list.size(); ++i)
        for (std::size_t j = 0; j < dim; ++i)
            vec_list[i][j] = point_list[i][j];
    
    return vec_list;
}


} // ParameterParser

#endif
