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
UniformVelocityField<dim>::UniformVelocityField(double zeta)
    : VelocityField<dim>("uniform", zeta)
{}

template <int dim>
UniformVelocityField<dim>::UniformVelocityField(dealii::Tensor<1, dim> v, double zeta)
    : VelocityField<dim>("uniform", zeta)
    , v(v)
{}



template <int dim>
UniformVelocityField<dim>::UniformVelocityField(std::map<std::string, boost::any> &am)
    : VelocityField<dim>("uniform", boost::any_cast<double>(am["coupling-constant"]))
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



template <int dim>
dealii::Tensor<1, dim> UniformVelocityField<dim>::
gradient(const dealii::Point<dim> &p, const unsigned int component) const
{
    return dealii::Tensor<1, dim>();
}



template <int dim>
void UniformVelocityField<dim>::
vector_gradient(const dealii::Point<dim> &p,
                std::vector<dealii::Tensor<1, dim>> &gradients) const
{
    for (auto &gradient : gradients)
        gradient = dealii::Tensor<1, dim>();
}



template <int dim>
void UniformVelocityField<dim>::
gradient_list(const std::vector<dealii::Point<dim>> &point_list,
              std::vector<dealii::Tensor<1, dim>> &gradients,
              const unsigned int component) const
{
    for (auto &gradient : gradients)
        gradient = dealii::Tensor<1, dim>();
}



template <int dim>
void UniformVelocityField<dim>::
vector_gradients(const std::vector<dealii::Point<dim>> &points,
                 std::vector<std::vector<dealii::Tensor<1, dim>>> &gradients) const
{
    for (auto &gradient_component : gradients)
        for (auto &gradient_component_at_point : gradient_component)
            gradient_component_at_point = dealii::Tensor<1, dim>();
}

template <int dim>
void UniformVelocityField<dim>::
vector_gradient_list(const std::vector<dealii::Point<dim>> &point_list,
                     std::vector<std::vector<dealii::Tensor<1, dim>>> &gradients) const
{
    for (auto &gradient_at_point : gradients)
        for (auto &gradient_component_at_point : gradient_at_point)
            gradient_component_at_point = dealii::Tensor<1, dim>();
}

template class UniformVelocityField<3>;
template class UniformVelocityField<2>;

// BOOST_CLASS_EXPORT_IMPLEMENT(UniformVelocityField<2>)
// BOOST_CLASS_EXPORT_IMPLEMENT(UniformVelocityField<3>)
