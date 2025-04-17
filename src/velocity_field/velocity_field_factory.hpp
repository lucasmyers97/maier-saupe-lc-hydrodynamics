#ifndef VELOCITY_FIELD_FACTORY_HPP
#define VELOCITY_FIELD_FACTORY_HPP

#include <deal.II/base/parameter_handler.h>

#include "velocity_field.hpp"
#include "Utilities/ParameterParser.hpp"

#include <boost/any.hpp>

#include <Parameters/toml.hpp>

#include <map>
#include <memory>
#include <string>
#include <stdexcept>

namespace VelocityFieldFactory
{
    template <int dim>
    std::map<std::string, boost::any>
    parse_parameters(const toml::table& table);

    template <int dim>
    std::unique_ptr<VelocityField<dim>>
    velocity_field_factory(std::map<std::string, boost::any> am);

} // namespace velocity_fields_factory

#endif
