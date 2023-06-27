#ifndef VECTOR_CONVERSION_HPP
#define VECTOR_CONVERSION_HPP

#include <deal.II/base/point.h>
#include <stdexcept>

namespace vector_conversion
{

template <typename T>
T convert(const std::vector<double>& vec);

template <typename T, typename S>
T convert(const S& vec);

}

#endif
