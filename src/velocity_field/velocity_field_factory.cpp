#include "velocity_field/velocity_field_factory.hpp"

#include <deal.II/base/parameter_handler.h>

#include "velocity_field/velocity_field.hpp"
#include "velocity_field/uniform_velocity_field.hpp"
#include "velocity_field/quadratic_velocity_field.hpp"
#include "Utilities/ParameterParser.hpp"
#include "Utilities/vector_conversion.hpp"

#include <boost/program_options.hpp>
#include <boost/any.hpp>

#include <map>
#include <memory>
#include <string>
#include <stdexcept>

namespace VelocityFieldFactory
{
    template <int dim>
    std::map<std::string, boost::any>
    parse_parameters(const toml::table& vf_table)
    {
        std::map<std::string, boost::any> vf_params;

        const auto name = vf_table["name"].value<std::string>();
        const auto zeta = vf_table["coupling-constant"].value<double>();

        if (!name) throw std::invalid_argument("No velocity field name provided in toml file");
        if (!zeta) throw std::invalid_argument("No velocity field coupling constant provided in toml file");

        vf_params["name"] = name.value();
        vf_params["coupling-constant"] = zeta.value();

        if (name.value() == "uniform")
        {
            if (!vf_table["velocity_vector"].is_array())
                throw std::invalid_argument("No velocity_vector array in toml file");
            const auto velocity_vector
                = toml::convert<std::vector<double>>(*vf_table["velocity_vector"].as_array());
            vf_params["velocity-vector"] = velocity_vector;
        } else if (name.value() == "quadratic")
        {
            if (!vf_table["endpoint_1"].is_array())
                throw std::invalid_argument("No endpoint_1 (velocity) array in toml file");
            const auto endpoint_1
                = toml::convert<std::vector<double>>(*vf_table["endpoint_1"].as_array());
            vf_params["endpoint-1"] = endpoint_1;

            if (!vf_table["endpoint_2"].is_array())
                throw std::invalid_argument("No endpoint_2 (velocity) array in toml file");
            const auto endpoint_2
                = toml::convert<std::vector<double>>(*vf_table["endpoint_2"].as_array());
            vf_params["endpoint-2"] = endpoint_2;

            if (!vf_table["flow_direction"].is_array())
                throw std::invalid_argument("No flow_direction (velocity) array in toml file");
            const auto flow_direction
                = toml::convert<std::vector<double>>(*vf_table["flow_direction"].as_array());
            vf_params["flow-direction"] = flow_direction;

            const auto max_flow_magnitude = vf_table["max_flow_magnitude"].value<double>();

            if (!max_flow_magnitude) throw std::invalid_argument("No max_flow_magnitude provided in toml file");
            vf_params["max-flow-magnitude"] = max_flow_magnitude.value();
        }
        else
            throw std::invalid_argument("Velocity flow name " 
                                        + name.value() 
                                        + " is not a valid class of flows");

        return vf_params;
    }

    template <int dim>
    std::unique_ptr<VelocityField<dim>>
    velocity_field_factory(std::map<std::string, boost::any> am)
    {
        std::string name = boost::any_cast<std::string>(am["name"]);

        if (name == "uniform")
        {
            if (am.empty())
                return std::make_unique<UniformVelocityField<dim>>();
            else
                return std::make_unique<UniformVelocityField<dim>>(am);
        } else if (name == "quadratic")
        {
            if (am.empty())
                return std::make_unique<QuadraticVelocityField<dim>>();
            else
                return std::make_unique<QuadraticVelocityField<dim>>(am);
        }
        else
        {
            throw std::invalid_argument("Invalid velocity field name in velocity_field_factory");
        }
    }

    template
    std::map<std::string, boost::any>
    parse_parameters<2>(const toml::table& table);

    template
    std::map<std::string, boost::any>
    parse_parameters<3>(const toml::table& table);

    template
    std::unique_ptr<VelocityField<2>>
    velocity_field_factory(std::map<std::string, boost::any> am);

    template
    std::unique_ptr<VelocityField<3>>
    velocity_field_factory(std::map<std::string, boost::any> am);

} // namespace VelocityFieldFactory
