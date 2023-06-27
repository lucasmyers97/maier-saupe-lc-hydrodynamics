#include "Utilities/vector_conversion.hpp"

namespace vector_conversion
{

template <>
dealii::Point<1> convert(const std::vector<double>& vec)
{
    if (vec.size() != 1)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Point");

    dealii::Point<1> p(vec[0]);
    return p;
}



template <>
dealii::Point<2> convert(const std::vector<double>& vec)
{
    if (vec.size() != 2)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Point");

    dealii::Point<2> p(vec[0], vec[1]);
    return p;
}



template <>
dealii::Point<3> convert(const std::vector<double>& vec)
{
    if (vec.size() != 3)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Point");

    dealii::Point<3> p(vec[0], vec[1], vec[2]);
    return p;
}



template <>
dealii::Tensor<1, 1> convert(const std::vector<double>& vec)
{
    if (vec.size() != 1)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Tensor");

    dealii::Tensor<1, 1> t;
    t[0] = vec[0];

    return t;
}



template <>
dealii::Tensor<1, 2> convert(const std::vector<double>& vec)
{
    if (vec.size() != 2)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Tensor");

    dealii::Tensor<1, 2> t;
    t[0] = vec[0];
    t[1] = vec[1];

    return t;
}



template <>
dealii::Tensor<1, 3> convert(const std::vector<double>& vec)
{
    if (vec.size() != 3)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Tensor");

    dealii::Tensor<1, 3> t;
    t[0] = vec[0];
    t[1] = vec[1];
    t[2] = vec[2];

    return t;
}



template <typename T, typename S>
T convert(const S& vec)
{
    T return_vec;
    return_vec.reserve(vec.size());

    for (const auto& item : vec)
        return_vec.push_back( convert<typename T::value_type>(item) );

    return return_vec;
}



template std::vector<dealii::Point<1>> convert(const std::vector<std::vector<double>>& vec);
template std::vector<dealii::Point<2>> convert(const std::vector<std::vector<double>>& vec);
template std::vector<dealii::Point<3>> convert(const std::vector<std::vector<double>>& vec);

template std::vector<dealii::Tensor<1, 1>> convert(const std::vector<std::vector<double>>& vec);
template std::vector<dealii::Tensor<1, 2>> convert(const std::vector<std::vector<double>>& vec);
template std::vector<dealii::Tensor<1, 3>> convert(const std::vector<std::vector<double>>& vec);

} // vector_conversion
