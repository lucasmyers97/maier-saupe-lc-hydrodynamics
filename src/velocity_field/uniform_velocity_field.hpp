#ifndef UNIFORM_VELOCITY_FIELD_HPP
#define UNIFORM_VELOCITY_FIELD_HPP

#include "velocity_field.hpp"

#include <boost/any.hpp>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <string>
#include <map>

template <int dim>
class UniformVelocityField : public VelocityField<dim>
{
public:
    UniformVelocityField();
    UniformVelocityField(dealii::Tensor<1, dim> v);
    UniformVelocityField(std::map<std::string, boost::any> &am);

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;


    virtual void vector_value(const dealii::Point<dim> &p,
					          dealii::Vector<double> &value) const override;

    virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                            std::vector<double> &value_list,
                            const unsigned int component = 0) const override;

    virtual void
    vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                      std::vector<dealii::Vector<double>>   &value_list)
                      const override;

private:
    dealii::Tensor<1, dim> v;
};

#endif
