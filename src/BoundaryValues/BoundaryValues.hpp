#ifndef BOUNDARY_VALUES_HPP
#define BOUNDARY_VALUES_HPP

#include "BoundaryValuesInterface.hpp"
#include <deal.II/base/function.h>
#include <string>
#include "maier_saupe_constants.hpp"

template<int dim>
class BoundaryValues : public dealii::Function<dim>, public BoundaryValuesInterface
{
public:
  virtual ~BoundaryValues() = default;
  const std::string name;

  BoundaryValues(std::string name_)
    : dealii::Function<dim>(maier_saupe_constants::vec_dim<dim>)
    , name(name_)
  {}
};



struct BoundaryValuesParams
{
  std::string name;
};


#endif
