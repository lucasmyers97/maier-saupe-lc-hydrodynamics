#ifndef VELOCITY_FIELD_HPP
#define VELOCITY_FIELD_HPP

#include <deal.II/base/function.h>

#include <string>


template<int dim>
class VelocityField : public dealii::Function<dim>
{
public:
    virtual ~VelocityField() = default;

    VelocityField(std::string name = std::string("uniform"))
        : dealii::Function<dim>(dim)
        , name(name)
    {}

    const std::string& return_name() const
    {
        return name;
    }

private:
    const std::string name;
};

#endif
