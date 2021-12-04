#ifndef BOUNDARY_VALUES_HPP
#define BOUNDARY_VALUES_HPP

#include "BoundaryValuesInterface.hpp"

#include <boost/serialization/access.hpp>

#include <deal.II/base/function.h>

#include <string>
#include "maier_saupe_constants.hpp"

template<int dim>
class BoundaryValues : public dealii::Function<dim>, public BoundaryValuesInterface
{
public:
  virtual ~BoundaryValues() = default;
  std::string name;

  BoundaryValues(std::string name_)
    : dealii::Function<dim>(maier_saupe_constants::vec_dim<dim>)
    , name(name_)
  {}

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & name;
    }
};

#endif
