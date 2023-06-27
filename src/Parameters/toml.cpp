#define TOML_IMPLEMENTATION
#include "Parameters/toml.hpp"

#include <vector>
#include <stdexcept>
#include <optional>

#include <deal.II/base/point.h>

template <>
std::vector<double> toml::convert(const toml::array& array)
{
    std::vector<double> return_vec;
    if (array.size() == 0)
        return return_vec;

    if (!array.is_homogeneous(toml::node_type::floating_point))
        throw std::invalid_argument("Array does not contain only doubles in "
                                    "conversion to std::vector<double>");

    return_vec.reserve(array.size());
    for (const auto &item : array)
        return_vec.push_back(item.value<double>().value());

    return return_vec;
}



template <typename T>
T toml::convert(const toml::array& array)
{
    T return_vec;
    if (array.size() == 0)
        return return_vec;

    if (!array.is_homogeneous(toml::node_type::array))
        throw std::invalid_argument("Array does not contain only arrays in "
                                    "conversion to std::vector<T>");

    return_vec.reserve(array.size());
    for (const auto& item : array)
        return_vec.push_back( convert<typename T::value_type>(*item.as_array()) );

    return return_vec;
}



template
std::vector<std::vector<double>> toml::convert(const toml::array& array);

template
std::vector<std::vector<std::vector<double>>> toml::convert(const toml::array& array);
