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
UniformConfiguration<dim>::UniformConfiguration(double S, double phi, double theta)
    : BoundaryValues<dim>("uniform")
    , S(S)
    , phi(phi)
    , theta(theta)
{}



template <int dim>
UniformConfiguration<dim>::UniformConfiguration(std::map<std::string, boost::any> &am)
    : BoundaryValues<dim>("uniform",
                          boost::any_cast<std::string>(am["boundary-condition"]))
    , S(boost::any_cast<double>(am["S-value"]))
    , phi(boost::any_cast<double>(am["phi"]))
    , theta(boost::any_cast<double>(am["theta"]))
{}



template <int dim>
UniformConfiguration<dim>::UniformConfiguration(po::variables_map vm)
  : BoundaryValues<dim>("uniform")
  , S(vm["S-value"].as<double>())
  , phi(vm["phi-value"].as<double>())
  , theta(vm["theta-value"].as<double>())
{}



template <int dim>
double UniformConfiguration<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
	double return_value = 0;
	switch (component){
	case 0:
		return_value = (std::sin(theta)*std::sin(theta) * std::cos(phi)*std::cos(phi) - 1.0/3.0);
		break;
	case 1:
		return_value = std::sin(theta)*std::sin(theta) * std::sin(phi)*std::cos(phi);
		break;
	case 2:
		return_value = 0.25 * (std::sin(phi + 2 * theta) - std::sin(phi - 2 * theta));
		break;
	case 3:
		return_value = (std::sin(theta)*std::sin(theta) * std::sin(phi)*std::sin(phi) - 1.0/3.0);
		break;
	case 4:
		return_value = 0.25 * (std::cos(phi - 2 * theta) - std::cos(phi + 2 * theta));
		break;
	}

	return S * return_value;
}



template <int dim>
void UniformConfiguration<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double>   &value) const
{
    value[0] = S * (std::sin(theta)*std::sin(theta) * std::cos(phi)*std::cos(phi) - 1.0/3.0);
    value[1] = S * std::sin(theta)*std::sin(theta) * std::sin(phi)*std::cos(phi);
    value[2] = S * 0.25 * (std::sin(phi + 2 * theta) - std::sin(phi - 2 * theta));
    value[3] = S * (std::sin(theta)*std::sin(theta) * std::sin(phi)*std::sin(phi) - 1.0/3.0);
    value[4] = S * 0.25 * (std::cos(phi - 2 * theta) - std::cos(phi + 2 * theta));
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
		    x = S * (std::sin(theta)*std::sin(theta) * std::cos(phi)*std::cos(phi) - 1.0/3.0);
		break;
	case 1:
        for (auto& x : value_list)
		    x = S * std::sin(theta)*std::sin(theta) * std::sin(phi)*std::cos(phi);
		break;
	case 2:
        for (auto& x : value_list)
		    x = S * 0.25 * (std::sin(phi + 2 * theta) - std::sin(phi - 2 * theta));
		break;
	case 3:
        for (auto& x : value_list)
		    x = S * (std::sin(theta)*std::sin(theta) * std::sin(phi)*std::sin(phi) - 1.0/3.0);
		break;
	case 4:
        for (auto& x : value_list)
		    x = S * 0.25 * (std::cos(phi - 2 * theta) - std::cos(phi + 2 * theta));
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
        value[0] = S * (std::sin(theta)*std::sin(theta) * std::cos(phi)*std::cos(phi) - 1.0/3.0);
        value[1] = S * std::sin(theta)*std::sin(theta) * std::sin(phi)*std::cos(phi);
        value[2] = S * 0.25 * (std::sin(phi + 2 * theta) - std::sin(phi - 2 * theta));
        value[3] = S * (std::sin(theta)*std::sin(theta) * std::sin(phi)*std::sin(phi) - 1.0/3.0);
        value[4] = S * 0.25 * (std::cos(phi - 2 * theta) - std::cos(phi + 2 * theta));
    }
}

template class UniformConfiguration<3>;
template class UniformConfiguration<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(UniformConfiguration<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(UniformConfiguration<3>)
