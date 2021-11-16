#include "UniformConfiguration.hpp"
#include <vector>
#include <cmath>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

template <int dim>
UniformConfiguration<dim>::UniformConfiguration(double S_, double phi_)
    : S(S_)
    , phi(phi_)
{}



template <int dim>
double UniformConfiguration<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
	double return_value = 0;
	switch (component){
	case 0:
		return_value = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
		break;
	case 1:
		return_value = 0.5 * S * std::sin(2*phi);
		break;
	case 2:
		return_value = 0.0;
		break;
	case 3:
		return_value = 0.5 * S * ( 1.0/3.0 - std::cos(2*phi) );
		break;
	case 4:
		return_value = -1.0/3.0;
		break;
	}

	return return_value;
}



template <int dim>
void UniformConfiguration<dim>::
vector_value(const dealii::Point<dim> &p, 
             dealii::Vector<double>   &value) const
{
	value[0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
	value[1] = 0.5 * S * std::sin(2*phi);
	value[2] = 0.0;
	value[3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*phi) );
	value[4] = -1.0/3.0;
}



template <int dim>
void UniformConfiguration<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
	switch (component){
	case 0:
        for (auto& x : value_list)
		    x = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
		break;
	case 1:
        for (auto& x : value_list)
		    x = 0.5 * S * std::sin(2*phi);
		break;
	case 2:
        for (auto& x : value_list)
		    x = 0.0;
		break;
	case 3:
        for (auto& x : value_list)
		    x = 0.5 * S * ( 1.0/3.0 - std::cos(2*phi) );
		break;
	case 4:
        for (auto& x : value_list)
		    x = -1.0/3.0;
		break;
	}
}



template <int dim>
void UniformConfiguration<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    for (auto& value : value_list)
    { 
	    value[0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
	    value[1] = 0.5 * S * std::sin(2*phi);
	    value[2] = 0.0;
	    value[3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*phi) );
	    value[4] = -1.0/3.0;
    }
}

template class UniformConfiguration<3>;
template class UniformConfiguration<2>;