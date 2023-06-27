#ifndef TOML_HPP
#define TOML_HPP

#define TOML_HEADER_ONLY 0
#include <tomlplusplus/toml.hpp>

#include <vector>

#include <deal.II/base/point.h>

namespace toml
{
    template <typename T>
    T convert(const array&);

} // toml

#endif
