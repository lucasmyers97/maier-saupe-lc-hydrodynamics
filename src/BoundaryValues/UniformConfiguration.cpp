#include "UniformConfiguration.hpp"
#include "BoundaryValues.hpp"

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/serialization/export.hpp>
#include <boost/any.hpp>

#include <vector>
#include <cmath>
#include <map>
#include <string>

namespace po = boost::program_options;

template <int dim>
UniformConfiguration<dim>::UniformConfiguration()
    : BoundaryValues<dim>("uniform")
{}

template <int dim>
UniformConfiguration<dim>::UniformConfiguration(double S_, double phi_)
    : S(S_)
    , phi(phi_)
    , BoundaryValues<dim>("uniform")
{}



template <int dim>
UniformConfiguration<dim>::UniformConfiguration(std::map<std::string, boost::any> &am)
    : S(boost::any_cast<double>(am["S-value"]))
    , phi(boost::any_cast<double>(am["phi"]))
    , BoundaryValues<dim>("uniform",
                          boost::any_cast<std::string>(am["boundary-condition"]))
{}



template <int dim>
UniformConfiguration<dim>::UniformConfiguration(po::variables_map vm)
  : S(vm["S-value"].as<double>())
  , phi(vm["phi-value"].as<double>())
  , BoundaryValues<dim>("uniform")
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
		return_value = 0;
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
	value[4] = 0;
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
		    x = 0;
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
	    value[4] = 0;
    }
}

template class UniformConfiguration<3>;
template class UniformConfiguration<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(UniformConfiguration<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(UniformConfiguration<3>)
