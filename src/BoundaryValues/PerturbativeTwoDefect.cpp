#include "PerturbativeTwoDefect.hpp"
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

namespace
{
    std::string return_defect_name(PerturbativeTwoDefectCharge charge)
    {
        switch (charge)
            {
            case PerturbativeTwoDefectCharge::plus_half:
                return "plus_half";
            case PerturbativeTwoDefectCharge::minus_half:
                return "minus_half";
            default:
                throw std::invalid_argument("Inputted incorrect charge");
            }
    }

    double return_defect_charge_val(PerturbativeTwoDefectCharge charge)
    {
        switch (charge)
            {
            case PerturbativeTwoDefectCharge::plus_half:
                return 0.5;
            case PerturbativeTwoDefectCharge::minus_half:
                return -0.5;
            default:
                throw std::invalid_argument("Inputted incorrect charge");
            }
    }

    PerturbativeTwoDefectCharge get_charge_from_name(const std::string &charge_name)
    {
        if (charge_name == "plus-half")
        {
            return PerturbativeTwoDefectCharge::plus_half;
        } else if (charge_name == "minus-half")
        {
            return PerturbativeTwoDefectCharge::minus_half;
        } else
        {
            throw std::invalid_argument("Inputted incorrect charge name");
        }
    }

    PerturbativeTwoDefectPosition get_position_from_name(const std::string &position_name)
    {
        if (position_name == "left")
            return PerturbativeTwoDefectPosition::left;
        else if (position_name == "right")
            return PerturbativeTwoDefectPosition::right;
        else
            throw std::invalid_argument("Inputted incorrect position name");
    }

    PerturbativeTwoDefectIsomorph get_isomorph_from_name(const std::string &isomorph_name)
    {
        if (isomorph_name == "a")
            return PerturbativeTwoDefectIsomorph::a;
        else if (isomorph_name == "b")
            return PerturbativeTwoDefectIsomorph::b;
        else
            throw std::invalid_argument("Inputted incorrect position name");
    }
}



template <int dim>
PerturbativeTwoDefect<dim>::PerturbativeTwoDefect()
    : BoundaryValues<dim>("plus-half")
    , k(return_defect_charge_val(charge))
{}



template <int dim>
PerturbativeTwoDefect<dim>::PerturbativeTwoDefect(double S_, PerturbativeTwoDefectCharge charge_)
    : BoundaryValues<dim>(return_defect_name(charge_))
    , S0(S_)
    , charge(charge_)
    , k(return_defect_charge_val(charge_)) {}



template <int dim>
PerturbativeTwoDefect<dim>::PerturbativeTwoDefect(std::map<std::string, boost::any> &am)
    : BoundaryValues<dim>(boost::any_cast<std::string>(am["defect-charge-name"]),
                          boost::any_cast<std::string>(am["boundary-condition"]))
    , S0(boost::any_cast<double>(am["S-value"]))
    , eps(boost::any_cast<double>(am["anisotropy-eps"]))
    , d(boost::any_cast<double>(am["defect-distance"]))
    , charge(get_charge_from_name(boost::any_cast<std::string>(am["defect-charge-name"])))
    , pos(get_position_from_name(boost::any_cast<std::string>(am["defect-position-name"])))
    , isomorph(get_isomorph_from_name(boost::any_cast<std::string>(am["defect-isomorph-name"])))
    , k(return_defect_charge_val(charge))
{
    std::vector<std::vector<double>> defect_coords 
        = boost::any_cast<std::vector<std::vector<double>>>(am["defect-positions"]);

    if (defect_coords.size() != 1)
        throw std::invalid_argument("Too many defect positions specified in "
                                    "parameters");

    for (std::size_t i = 0; i < defect_coords[0].size(); ++i)
        center[i] = defect_coords[0][i];

    this->defect_pts.push_back(center);

    // isolated part of pertubration changes sign on left vs right position
    // because of change in defect orientation
    if (pos == PerturbativeTwoDefectPosition::left)
        isolated_sign = 1.0;
    else
        isolated_sign = -1.0;

    // isomorph changes sign of isolated and interacting perturbation
    if (isomorph == PerturbativeTwoDefectIsomorph::a)
        isomorph_sign = 1.0;
    else
        isomorph_sign = -1.0;
}



template <int dim>
double PerturbativeTwoDefect<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
    const auto q = p - center;

    const double r = q.norm();
    const double theta = std::atan2(q[1], q[0]);
    const double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);

    const double phi = calc_phi(r, theta);

	switch (component)
	{
	case 0:
		return 0.5 * S * ( 1.0/3.0 + std::cos(2 * phi) );
	case 1:
		return 0.5 * S * std::sin(2 * phi);
	case 2:
		return 0.0;
	case 3:
		return 0.5 * S * ( 1.0/3.0 - std::cos(2 * phi) );
	case 4:
		return 0.0;
    default:
        throw std::invalid_argument("Incorrect component argument");
	}
}



template <int dim>
void PerturbativeTwoDefect<dim>::
vector_value(const dealii::Point<dim> &p, 
             dealii::Vector<double>   &value) const
{
    const auto q = p - center;

    const double r = q.norm();
    const double theta = std::atan2(q[1], q[0]);
    const double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);

    double phi = calc_phi(r, theta);

	value[0] = 0.5 * S * ( 1.0/3.0 + std::cos(2 * phi) );
	value[1] = 0.5 * S * std::sin(2 * phi);
	value[2] = 0.0;
	value[3] = 0.5 * S * ( 1.0/3.0 - std::cos(2 * phi) );
	value[4] = 0.0;
}



template <int dim>
void PerturbativeTwoDefect<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        const auto q = point_list[i] - center;

        const double r = q.norm();
        const double theta = std::atan2(q[1], q[0]);
        const double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
        const double phi = calc_phi(r, theta);

	    switch (component)
	    {
	    case 0:
	    	value_list[i] = 0.5 * S * ( 1.0/3.0 + std::cos(2 * phi) );
	    	break;
	    case 1:
	    	value_list[i] = 0.5 * S * std::sin(2 * phi);
	    	break;
	    case 2:
	    	value_list[i] = 0.0;
	    	break;
	    case 3:
	        value_list[i] = 0.5 * S * ( 1.0/3.0 - std::cos(2 * phi) );
	    	break;
	    case 4:
	    	value_list[i] = 0.0;
	    	break;
	    }
    }
}



template <int dim>
void PerturbativeTwoDefect<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    for (std::size_t i = 0; i < point_list.size(); ++i)
    { 
        const auto q = point_list[i] - center;

        const double r = q.norm();
        const double theta = std::atan2(q[1], q[0]);
        const double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
        const double phi = calc_phi(r, theta);

	    value_list[i][0] = 0.5 * S * ( 1.0/3.0 + std::cos(2 * phi) );
	    value_list[i][1] = 0.5 * S * std::sin(2 * phi);
	    value_list[i][2] = 0.0;
	    value_list[i][3] = 0.5 * S * ( 1.0/3.0 - std::cos(2 * phi) );
	    value_list[i][4] = 0.0;
    }
}



template <int dim>
double PerturbativeTwoDefect<dim>::
calc_phi(const double r, const double theta) const
{
    // isotropic elasticity contribution
    double phi_iso = k * theta;
    if (pos == PerturbativeTwoDefectPosition::left)
        phi_iso += -k * std::atan2(r * std::sin(theta), r * std::cos(theta) - d);
    else
        phi_iso += -k * std::atan2(r * std::sin(theta), r * std::cos(theta) + d);

    if (isomorph == PerturbativeTwoDefectIsomorph::a)
        phi_iso += 0.5 * M_PI;

    // perturbative contribution from anisotropic elasticity
    double perturbation = 0.0;
    if (charge == PerturbativeTwoDefectCharge::plus_half)
    {
        // isolated perturbation
        perturbation += 3.0 * eps * std::sin(theta) / 4.0;
        perturbation *= isolated_sign;

        // interacting perturbation
        perturbation += -eps * r * std::sin(2 * theta) / (8.0 * d);
        perturbation *= isomorph_sign;
    }
    else
    {
        // isolated perturbation
        perturbation += -5.0 * eps * std::sin(3 * theta) / 36.0;
        perturbation *= isolated_sign;

        // interacting perturbation
        perturbation += eps * r / (36.0 * d) * (std::sin(2 * theta) - std::sin(4 * theta));
        perturbation *= isomorph_sign;
    }

    return phi_iso + perturbation;
}


template class PerturbativeTwoDefect<3>;
template class PerturbativeTwoDefect<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(PerturbativeTwoDefect<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(PerturbativeTwoDefect<3>)
