#ifndef BOUNDARY_VALUES_HPP
#define BOUNDARY_VALUES_HPP

#include <deal.II/base/function.h>
#include <string>

template<int dim>
class BoundaryValues : public dealii::Function<dim>
{
public:
    virtual ~BoundaryValues() = default;
    std::string name = "";
};


#endif