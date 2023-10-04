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

    template <int dim>
    typename DefectConfiguration<dim>::DefectAxis get_axis_from_name(const std::string &axis_name)
    {
        if (axis_name == "x")
            return DefectConfiguration<dim>::DefectAxis::x;
        else if (axis_name == "y")
            return DefectConfiguration<dim>::DefectAxis::y;
        else if (axis_name == "z")
            return DefectConfiguration<dim>::DefectAxis::z;
        else
            throw std::invalid_argument("Axis name in DefectConfiguration must be x, y, or z");
    }

    template <int dim>
    double calc_phi(const dealii::Point<dim> &p, const typename DefectConfiguration<dim>::DefectAxis axis);

    template <>
    double calc_phi<3>(const dealii::Point<3> &p, const typename DefectConfiguration<3>::DefectAxis axis)
    {
        if (axis == DefectConfiguration<3>::DefectAxis::x)
	        return std::atan2(p[2], p[1]);
        else if (axis == DefectConfiguration<3>::DefectAxis::y)
	        return std::atan2(p[0], p[2]);
        else if (axis == DefectConfiguration<3>::DefectAxis::z)
	        return std::atan2(p[1], p[0]);
        else
            throw std::invalid_argument("Axis name in DefectConfiguration must be x, y, or z");
    }

    template <>
    double calc_phi<2>(const dealii::Point<2> &p, const typename DefectConfiguration<2>::DefectAxis axis)
    {
        if (axis == DefectConfiguration<2>::DefectAxis::x)
	        return std::atan2(0, p[1]);
        else if (axis == DefectConfiguration<2>::DefectAxis::y)
	        return std::atan2(p[0], 0);
        else if (axis == DefectConfiguration<2>::DefectAxis::z)
	        return std::atan2(p[1], p[0]);
        else
            throw std::invalid_argument("Axis name in DefectConfiguration must be x, y, or z");
    }


    template <int dim>
    double calc_r(const dealii::Point<dim> &p, const typename DefectConfiguration<dim>::DefectAxis axis);

    template <>
    double calc_r<3>(const dealii::Point<3> &p, const typename DefectConfiguration<3>::DefectAxis axis)
    {
        if (axis == DefectConfiguration<3>::DefectAxis::x)
	        return std::sqrt(p[2]*p[2] + p[1]*p[1]);
        else if (axis == DefectConfiguration<3>::DefectAxis::y)
	        return std::sqrt(p[2]*p[2] + p[0]*p[0]);
        else if (axis == DefectConfiguration<3>::DefectAxis::z)
	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        else
            throw std::invalid_argument("Axis name in DefectConfiguration must be x, y, or z");
    }
    
    template <>
    double calc_r<2>(const dealii::Point<2> &p, const typename DefectConfiguration<2>::DefectAxis axis)
    {
        if (axis == DefectConfiguration<2>::DefectAxis::x)
	        return std::abs(p[1]);
        else if (axis == DefectConfiguration<2>::DefectAxis::y)
	        return std::abs(p[0]);
        else if (axis == DefectConfiguration<2>::DefectAxis::z)
	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        else
            throw std::invalid_argument("Axis name in DefectConfiguration must be x, y, or z");
    }

    template <int dim>
    dealii::Point<3> calc_n(double phi, const typename DefectConfiguration<dim>::DefectAxis axis);

    template <>
    dealii::Point<3> calc_n<3>(double theta, const typename DefectConfiguration<3>::DefectAxis axis)
    {
        if (axis == DefectConfiguration<3>::DefectAxis::x)
	        return dealii::Point<3>({0, std::cos(theta), std::sin(theta)});
        else if (axis == DefectConfiguration<3>::DefectAxis::y)
	        return dealii::Point<3>({std::sin(theta), 0, std::cos(theta)});
        else if (axis == DefectConfiguration<3>::DefectAxis::z)
	        return dealii::Point<3>({std::cos(theta), std::sin(theta), 0});
        else
            throw std::invalid_argument("Axis name in DefectConfiguration must be x, y, or z");
    }

    template <>
    dealii::Point<3> calc_n<2>(double theta, const typename DefectConfiguration<2>::DefectAxis axis)
    {
        if (axis == DefectConfiguration<2>::DefectAxis::x)
	        return dealii::Point<3>({0, std::cos(theta), std::sin(theta)});
        else if (axis == DefectConfiguration<2>::DefectAxis::y)
	        return dealii::Point<3>({std::sin(theta), 0, std::cos(theta)});
        else if (axis == DefectConfiguration<2>::DefectAxis::z)
	        return dealii::Point<3>({std::cos(theta), std::sin(theta), 0});
        else
            throw std::invalid_argument("Axis name in DefectConfiguration must be x, y, or z");
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
    , axis(get_axis_from_name<dim>(boost::any_cast<std::string>(am["defect-axis"])))
{
    std::vector<std::vector<double>> defect_coords 
        = boost::any_cast<std::vector<std::vector<double>>>(am["defect-positions"]);

    if (defect_coords.size() != 1)
        throw std::invalid_argument("Too many defect positions specified in "
                                    "parameters");

    for (std::size_t i = 0; i < defect_coords[0].size(); ++i)
        center[i] = defect_coords[0][i];

    this->defect_pts.push_back(center);
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
	const double phi = calc_phi(p, axis);
    const double r = calc_r(p, axis);
    const double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
    const auto n = calc_n<dim>(k * phi, axis);

	switch (component)
	{
	case 0:
        return S * (n[0]*n[0] - 1.0/3.0);
	case 1:
        return S * n[0]*n[1];
	case 2:
        return S * n[0]*n[2];
	case 3:
        return S * (n[1]*n[1] - 1.0/3.0);
	case 4:
        return S * n[1]*n[2];
    default:
        throw std::invalid_argument("In DefectConfiguration::value `component` must be 0 to 4");
	}
}



template <int dim>
void DefectConfiguration<dim>::
vector_value(const dealii::Point<dim> &p, 
             dealii::Vector<double>   &value) const
{
	double phi = calc_phi(p, axis);
    double r = calc_r(p, axis);
    double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
    const auto n = calc_n<dim>(k * phi, axis);

	value[0] = S * (n[0]*n[0] - 1.0/3.0);
	value[1] = S * n[0]*n[1];
	value[2] = S * n[0]*n[2];
	value[3] = S * (n[1]*n[1] - 1.0/3.0);
	value[4] = S * n[1]*n[2];
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
    dealii::Point<3> n;

    for (std::size_t i = 0; i < point_list.size(); ++i)
	{
        r = calc_r(point_list[i], axis);
        S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
		phi = calc_phi(point_list[i], axis);
        n = calc_n<dim>(k * phi, axis);
	    switch (component)
        {
	    case 0:
            value_list[i] = S * (n[0]*n[0] - 1.0/3.0);
            break;
	    case 1:
            value_list[i] = S * n[0]*n[1];
            break;
	    case 2:
            value_list[i] = S * n[0]*n[2];
            break;
	    case 3:
            value_list[i] = S * (n[1]*n[1] - 1.0/3.0);
            break;
	    case 4:
            value_list[i] = S * n[1]*n[2];
            break;
        }
    }
}



/** DIMENSIONALLY-WEIRD projects distances + angles into x-y plane */
template <int dim>
void DefectConfiguration<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
	double phi = 0;
    double r = 0;
    double S = 0;
    dealii::Point<3> n;
    for (std::size_t i = 0; i < point_list.size(); ++i)
    { 
		phi = calc_phi(point_list[i], axis);
        r = calc_r(point_list[i], axis);
        S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
        n = calc_n<dim>(k * phi, axis);

	    value_list[i][0] = S * (n[0]*n[0] - 1.0/3.0);
	    value_list[i][1] = S * n[0]*n[1];
	    value_list[i][2] = S * n[0]*n[2];
	    value_list[i][3] = S * (n[1]*n[1] - 1.0/3.0);
	    value_list[i][4] = S * n[1]*n[2];
    }
}

template class DefectConfiguration<3>;
template class DefectConfiguration<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(DefectConfiguration<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(DefectConfiguration<3>)
