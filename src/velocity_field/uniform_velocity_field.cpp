#include "velocity_field.hpp"
#include "uniform_velocity_field.hpp"
#include "Utilities/vector_conversion.hpp"

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/serialization/export.hpp>
#include <boost/any.hpp>

#include <vector>
#include <cmath>
#include <map>
#include <string>

template <int dim>
UniformVelocityField<dim>::UniformVelocityField()
    : VelocityField<dim>("uniform")
{}

template <int dim>
UniformVelocityField<dim>::UniformVelocityField(dealii::Tensor<1, dim> v)
    : VelocityField<dim>("uniform")
    , v(v)
{}



template <int dim>
UniformVelocityField<dim>::UniformVelocityField(std::map<std::string, boost::any> &am)
    : VelocityField<dim>("uniform")
    , v(vector_conversion::convert<dealii::Tensor<1, dim>>(
            boost::any_cast<std::vector<double>>(am["velocity-vector"])
            )
        )
{}



template <int dim>
double UniformVelocityField<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
	return v[component];
}



template <int dim>
void UniformVelocityField<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double>   &value) const
{
    for (std::size_t i = 0; i < dim; ++i)
        value[i] = v[i];
}



template <int dim>
void UniformVelocityField<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
    for (auto& x : value_list)
        x = v[component];
}



template <int dim>
void UniformVelocityField<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    for (auto& value : value_list)
        for (std::size_t i = 0; i < dim; ++i)
            value[i] = v[i];
}

template class UniformVelocityField<3>;
template class UniformVelocityField<2>;

// BOOST_CLASS_EXPORT_IMPLEMENT(UniformVelocityField<2>)
// BOOST_CLASS_EXPORT_IMPLEMENT(UniformVelocityField<3>)
