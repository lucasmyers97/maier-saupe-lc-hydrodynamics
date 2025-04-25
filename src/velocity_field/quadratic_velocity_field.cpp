#include "velocity_field.hpp"
#include "quadratic_velocity_field.hpp"
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
QuadraticVelocityField<dim>::QuadraticVelocityField(double zeta)
    : VelocityField<dim>("quadratic", zeta)
{}

template <int dim>
QuadraticVelocityField<dim>::
QuadraticVelocityField(dealii::Tensor<1, dim> a,
                       dealii::Tensor<1, dim> b,
                       double max_flow_magnitude,
                       dealii::Tensor<1, dim> u,
                       double zeta)
    : VelocityField<dim>("quadratic", zeta)
    , a(a)
    , b(b)
    , length((a - b).norm())
    , max_flow_magnitude(max_flow_magnitude)
    , u(u)
{}



template <int dim>
QuadraticVelocityField<dim>::QuadraticVelocityField(std::map<std::string, boost::any> &am)
    : VelocityField<dim>("quadratic", boost::any_cast<double>(am["coupling-constant"]))
    , a(vector_conversion::convert<dealii::Tensor<1, dim>>(
            boost::any_cast<std::vector<double>>(am["endpoint-1"])
            )
        )
    , b(vector_conversion::convert<dealii::Tensor<1, dim>>(
            boost::any_cast<std::vector<double>>(am["endpoint-2"])
            )
        )
    , length((a - b).norm())
    , max_flow_magnitude(boost::any_cast<double>(am["max-flow-magnitude"]))
    , u(vector_conversion::convert<dealii::Tensor<1, dim>>(
            boost::any_cast<std::vector<double>>(am["flow-direction"])
            )
        )
{}



template <int dim>
double QuadraticVelocityField<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
    auto proj = (a - p) * (a - b) / length;
    
    auto flow_magnitude = proj * (length - proj) 
                          * 4 / (length * length) 
                          * max_flow_magnitude;
    
    return u[component] * flow_magnitude;
}



template <int dim>
void QuadraticVelocityField<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double>   &value) const
{
    auto proj = (a - p) * (a - b) / length;
    
    auto flow_magnitude = proj * (length - proj) 
                          * 4 / (length * length) 
                          * max_flow_magnitude;
    
    for (std::size_t i = 0; i < dim; ++i)
        value[i] = u[i] * flow_magnitude;
}



template <int dim>
void QuadraticVelocityField<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
    for (std::size_t i = 0; i < point_list.size(); ++i) 
    {
        auto proj = (a - point_list[i]) * (a - b) / length;
        
        auto flow_magnitude = proj * (length - proj) 
                              * 4 / (length * length) 
                              * max_flow_magnitude;
        
        value_list[i] = u[component] * flow_magnitude;
    }
}



template <int dim>
void QuadraticVelocityField<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    for (std::size_t i = 0; i < point_list.size(); ++i) 
    {
        auto proj = (a - point_list[i]) * (a - b) / length;
        
        auto flow_magnitude = proj * (length - proj) 
                              * 4 / (length * length) 
                              * max_flow_magnitude;
        
        for (std::size_t j = 0; j < dim; ++j)
            value_list[i][j] = u[j] * flow_magnitude;
    }
}



template <int dim>
dealii::Tensor<1, dim> QuadraticVelocityField<dim>::
gradient(const dealii::Point<dim> &p, const unsigned int component) const
{
    return (
        2 * (a - p) * (a - b) / (length * length)
        - 1
        ) * (a - b) 
        * 4 * max_flow_magnitude / (length * length)
        * u[component];
}



template <int dim>
void QuadraticVelocityField<dim>::
vector_gradient(const dealii::Point<dim> &p,
                std::vector<dealii::Tensor<1, dim>> &gradients) const
{
    for (std::size_t i = 0; i < dim; ++i)
        gradients[i] = (
            2 * (a - p) * (a - b) / (length * length)
            - 1
            ) * (a - b) 
            * 4 * max_flow_magnitude / (length * length)
            * u[i];
}



template <int dim>
void QuadraticVelocityField<dim>::
gradient_list(const std::vector<dealii::Point<dim>> &point_list,
              std::vector<dealii::Tensor<1, dim>> &gradients,
              const unsigned int component) const
{
    for (std::size_t i = 0; i < point_list.size(); ++i)
        gradients[i] = (
            2 * (a - point_list[i]) * (a - b) / (length * length)
            - 1
            ) * (a - b) 
            * 4 * max_flow_magnitude / (length * length)
            * u[component];
}



template <int dim>
void QuadraticVelocityField<dim>::
vector_gradients(const std::vector<dealii::Point<dim>> &points,
                 std::vector<std::vector<dealii::Tensor<1, dim>>> &gradients) const
{
    // c for component
    for (std::size_t c = 0; c < dim; ++c)
        for (std::size_t i = 0; i < points.size(); ++i)
            gradients[c][i] = (
                2 * (a - points[i]) * (a - b) / (length * length)
                - 1
                ) * (a - b) 
                * 4 * max_flow_magnitude / (length * length)
                * u[c];
}

template <int dim>
void QuadraticVelocityField<dim>::
vector_gradient_list(const std::vector<dealii::Point<dim>> &point_list,
                     std::vector<std::vector<dealii::Tensor<1, dim>>> &gradients) const
{
    for (std::size_t i = 0; i < point_list.size(); ++i)
        for (std::size_t c = 0; c < dim; ++c) // c for component
            gradients[i][c] = (
                2 * (a - point_list[i]) * (a - b) / (length * length)
                - 1
                ) * (a - b) 
                * 4 * max_flow_magnitude / (length * length)
                * u[c];
}

template class QuadraticVelocityField<3>;
template class QuadraticVelocityField<2>;

// BOOST_CLASS_EXPORT_IMPLEMENT(QuadraticVelocityField<2>)
// BOOST_CLASS_EXPORT_IMPLEMENT(QuadraticVelocityField<3>)
