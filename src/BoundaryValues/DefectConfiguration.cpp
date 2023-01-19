#include "DefectConfiguration.hpp"
#include "Utilities/maier_saupe_constants.hpp"
#include <string>
#include <boost/serialization/export.hpp>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/any.hpp>

#include <stdexcept>
#include <map>
#include <string>

namespace po = boost::program_options;

namespace
{
    std::string return_defect_name(DefectCharge charge)
    {
        switch (charge)
            {
            case DefectCharge::plus_half:
                return "plus_half";
            case DefectCharge::minus_half:
                return "minus_half";
            case DefectCharge::plus_one:
                return "plus_one";
            case DefectCharge::minus_one:
                return "minus_one";
            default:
                throw std::invalid_argument("Inputted incorrect charge");
            }
    }

    double return_defect_charge_val(DefectCharge charge)
    {
        switch (charge)
            {
            case DefectCharge::plus_half:
                return 0.5;
            case DefectCharge::minus_half:
                return -0.5;
            case DefectCharge::plus_one:
                return 1.0;
            case DefectCharge::minus_one:
                return -1.0;
            default:
                throw std::invalid_argument("Inputted incorrect charge");
            }
    }

    DefectCharge get_charge_from_name(const std::string charge_name)
    {
        if (charge_name == "plus-half")
        {
            return DefectCharge::plus_half;
        } else if (charge_name == "minus-half")
        {
            return DefectCharge::minus_half;
        }
        else if (charge_name == "plus-one")
        {
            return DefectCharge::plus_one;
        } else if (charge_name == "minus-one")
        {
            return DefectCharge::minus_one;
        } else
        {
            throw std::invalid_argument("Inputted incorrect charge name");
        }
    }
}



template <int dim>
DefectConfiguration<dim>::DefectConfiguration()
    : BoundaryValues<dim>("plus-half")
    , k(return_defect_charge_val(charge))
{}



template <int dim>
DefectConfiguration<dim>::DefectConfiguration(double S_, DefectCharge charge_)
    : BoundaryValues<dim>(return_defect_name(charge_))
    , S0(S_)
    , charge(charge_)
    , k(return_defect_charge_val(charge_)) {}



template <int dim>
DefectConfiguration<dim>::DefectConfiguration(std::map<std::string, boost::any> &am)
    : BoundaryValues<dim>(boost::any_cast<std::string>(am["defect-charge-name"]),
                          boost::any_cast<std::string>(am["boundary-condition"]))
    , S0(boost::any_cast<double>(am["S-value"]))
    , charge(get_charge_from_name(boost::any_cast<std::string>(am["defect-charge-name"])))
    , k(return_defect_charge_val(charge))
{
    std::vector<std::vector<double>> defect_coords 
        = boost::any_cast<std::vector<std::vector<double>>>(am["defect-positions"]);

    if (defect_coords.size() != 1)
        throw std::invalid_argument("Too many defect positions specified in "
                                    "parameters");

    for (std::size_t i = 0; i < defect_coords[0][i]; ++i)
        center[i] = defect_coords[0][i];
}



template <int dim>
DefectConfiguration<dim>::DefectConfiguration(po::variables_map vm)
    : BoundaryValues<dim>(vm["defect-charge-name"].as<std::string>())
    , S0(vm["S-value"].as<double>())
    , charge(get_charge_from_name(vm["defect-charge-name"].as<std::string>()))
    , k(return_defect_charge_val(charge))
{}



template <int dim>
double DefectConfiguration<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
	double phi = std::atan2(p[1], p[0]);
    double r = std::sqrt(p[0]*p[0] + p[1]*p[1]);
    double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
	double return_value = 0;

	switch (component)
	{
	case 0:
		return_value = 0.5 * S * ( 1.0/3.0 + std::cos(2*k*phi) );
		break;
	case 1:
		return_value = 0.5 * S * std::sin(2*k*phi);
		break;
	case 2:
		return_value = 0.0;
		break;
	case 3:
		return_value = 0.5 * S * ( 1.0/3.0 - std::cos(2*k*phi) );
		break;
	case 4:
		return_value = 0.0;
		break;
	}

	return return_value;
}



template <int dim>
void DefectConfiguration<dim>::
vector_value(const dealii::Point<dim> &p, 
             dealii::Vector<double>   &value) const
{
	double phi = std::atan2(p[1], p[0]);
    double r = std::sqrt(p[0]*p[0] + p[1]*p[1]);
    double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);

	value[0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*k*phi) );
	value[1] = 0.5 * S * std::sin(2*k*phi);
	value[2] = 0.0;
	value[3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*k*phi) );
	value[4] = 0.0;
}



template <int dim>
void DefectConfiguration<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
	double phi = 0;
    double r = 0;
    double S = 0;
	switch (component)
	{
	case 0:
        for (std::size_t i = 0; i < point_list.size(); ++i)
		{
            r = std::sqrt(point_list[i][0]*point_list[i][0] 
                          + point_list[i][1]*point_list[i][1]);
            S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
			phi = std::atan2(point_list[i][1], point_list[i][0]);
		    value_list[i] = 0.5 * S * ( 1.0/3.0 + std::cos(2*k*phi) );
		}
		break;
	case 1:
        for (std::size_t i = 0; i < point_list.size(); ++i)
		{
            r = std::sqrt(point_list[i][0]*point_list[i][0] 
                          + point_list[i][1]*point_list[i][1]);
            S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
			phi = std::atan2(point_list[i][1], point_list[i][0]);
		    value_list[i] = 0.5 * S * std::sin(2*k*phi);
		}
		break;
	case 2:
        for (std::size_t i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.0;
		break;
	case 3:
        for (std::size_t i = 0; i < point_list.size(); ++i)
		{
            r = std::sqrt(point_list[i][0]*point_list[i][0] 
                          + point_list[i][1]*point_list[i][1]);
            S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
			phi = std::atan2(point_list[i][1], point_list[i][0]);
		    value_list[i] = 0.5 * S * ( 1.0/3.0 - std::cos(2*k*phi) );
		}
		break;
	case 4:
        for (std::size_t i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.0;
		break;
	}
}



template <int dim>
void DefectConfiguration<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
	double phi = 0;
    double r = 0;
    double S = 0;
    for (std::size_t i = 0; i < point_list.size(); ++i)
    { 
		phi = std::atan2(point_list[i][1], point_list[i][0]);
        r = std::sqrt(point_list[i][0]*point_list[i][0] 
                      + point_list[i][1]*point_list[i][1]);
        S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);

	    value_list[i][0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*k*phi) );
	    value_list[i][1] = 0.5 * S * std::sin(2*k*phi);
	    value_list[i][2] = 0.0;
	    value_list[i][3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*k*phi) );
	    value_list[i][4] = 0.0;
    }
}

template class DefectConfiguration<3>;
template class DefectConfiguration<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(DefectConfiguration<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(DefectConfiguration<3>)
