#ifndef BOUNDARY_VALUES_FACTORY_HPP
#define BOUNDARY_VALUES_FACTORY_HPP

#include "BoundaryValuesInterface.hpp"
#include "BoundaryValues.hpp"
#include "DefectConfiguration.hpp"
#include "TwoDefectConfiguration.hpp"
#include "UniformConfiguration.hpp"

#include <boost/program_options.hpp>
#include <boost/any.hpp>

#include <map>
#include <memory>
#include <string>
#include <stdexcept>

namespace BoundaryValuesFactory
{
    namespace po = boost::program_options;

    template <int dim>
    std::unique_ptr<BoundaryValues<dim>>
    BoundaryValuesFactory(const po::variables_map &vm)
    {
        std::string name = vm["boundary-values-name"].as<std::string>();

        if (name == "uniform")
        {
            return std::make_unique<UniformConfiguration<dim>>(vm);
        }
        else if (name == "defect")
        {
            return std::make_unique<DefectConfiguration<dim>>(vm);
        }
        else if (name == "two-defect")
        {
            return std::make_unique<TwoDefectConfiguration<dim>>(vm);
        }
        else
        {
            throw std::invalid_argument("Invalid boundary value name in BoundaryValuesFactory");
        }
    }


    template <int dim>
    std::unique_ptr<BoundaryValues<dim>>
    BoundaryValuesFactory(const std::string name,
                          std::map<std::string, boost::any> am)
    {
        if (name == "uniform")
        {
            if (am.empty())
                return std::make_unique<UniformConfiguration<dim>>();
            else
                return std::make_unique<UniformConfiguration<dim>>(am);
        }
        else if (name == "defect")
        {
            if (am.empty())
                return std::make_unique<DefectConfiguration<dim>>();
            else
                return std::make_unique<DefectConfiguration<dim>>(am);
        }
        else if (name == "two-defect")
        {
            if (am.empty())
              return std::make_unique<TwoDefectConfiguration<dim>>();
            else
                return std::make_unique<TwoDefectConfiguration<dim>>(am);
        }
        else
        {
            throw std::invalid_argument("Invalid boundary value name in BoundaryValuesFactory");
        }
    }
} // namespace BoundaryValuesFactory

#endif
