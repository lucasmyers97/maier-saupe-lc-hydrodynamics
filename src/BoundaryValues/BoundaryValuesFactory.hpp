#ifndef BOUNDARY_VALUES_FACTORY_HPP
#define BOUNDARY_VALUES_FACTORY_HPP

#include "BoundaryValuesInterface.hpp"
#include "BoundaryValues.hpp"
#include "DefectConfiguration.hpp"
#include "UniformConfiguration.hpp"

#include <boost/program_options.hpp>
#include <memory>
#include <string>

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
  }
} // namespace BoundaryValuesFactory

#endif
