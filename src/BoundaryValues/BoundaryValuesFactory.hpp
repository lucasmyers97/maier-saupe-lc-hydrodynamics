#ifndef BOUNDARY_VALUES_FACTORY_HPP
#define BOUNDARY_VALUES_FACTORY_HPP

#include "BoundaryValuesInterface.hpp"
#include "BoundaryValues.hpp"
#include "DefectConfiguration.hpp"
#include "UniformConfiguration.hpp"

#include <memory>

namespace BoundaryValuesFactory
{
  std::unique_ptr<BoundaryValuesInterface>
  BoundaryValuesFactory(std::unique_ptr<BoundaryValueParams> params, int dim)
  {
    std::string name = (*params).name;

    if (dim == 2)
      {
        if (name == "uniform")
          {
            return std::make_unique<UniformConfiguration<2>>(*params);
          }
        else if (name == "defect")
          {
            return std::make_unique<DefectConfiguration<2>>(*params);
          }
      }
    else if (dim == 3)
      {
        if (name == "uniform")
          {
            return std::make_unique<UniformConfiguration<3>>(*params);
          }
        else if (name == "defect")
          {
            return std::make_unique<DefectConfiguration<3>>(*params);
          }
      }
  }
}

#endif
