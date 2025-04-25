#ifndef VELOCITY_FIELD_HPP
#define VELOCITY_FIELD_HPP

#include <deal.II/base/function.h>

#include <string>


template<int dim>
class VelocityField : public dealii::Function<dim>
{
public:
    virtual ~VelocityField() = default;

    VelocityField(std::string name, double zeta = 0.0)
        : dealii::Function<dim>(dim)
        , name(name)
        , zeta(zeta)
    {}

    const std::string& return_name() const
    {
        return name;
    }

    double return_coupling_constant() const
    {
        return zeta;
    }

    void set_coupling_constant(double zeta_)
    {
        zeta = zeta_;
    }

private:
    const std::string name;
    double zeta;
};

#endif
