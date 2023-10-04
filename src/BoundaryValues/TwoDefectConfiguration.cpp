#include "TwoDefectConfiguration.hpp"
#include "Utilities/maier_saupe_constants.hpp"
#include <deal.II/base/numbers.h>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/serialization/export.hpp>
#include <boost/any.hpp>

#include <stdexcept>
#include <cmath>
#include <map>
#include <string>

namespace po = boost::program_options;

namespace
{
    std::string return_defect_name(TwoDefectCharge charge)
    {
        switch (charge)
        {
        case TwoDefectCharge::plus_half_minus_half:
            return "plus_half_minus_half";
        case TwoDefectCharge::plus_half_minus_half_alt:
            return "plus_half_minus_half_alt";
        case TwoDefectCharge::plus_half_plus_half:
            return "plus_half_plus_half";
        default:
            throw std::invalid_argument("Inputted incorrect charge for two defect configuration");
        }
    }

    std::vector<double> return_defect_charge_val(TwoDefectCharge charge)
    {
        switch (charge)
        {
        case TwoDefectCharge::plus_half_minus_half:
            return {0.5, -0.5};
        case TwoDefectCharge::plus_half_minus_half_alt:
            return {0.5, -0.5};
        case TwoDefectCharge::plus_half_plus_half:
            return {0.5, 0.5};
        default:
            throw std::invalid_argument("Inputted incorrect charge for two defect configuration");
        }
    }

    TwoDefectCharge get_charge_from_name(const std::string charge_name)
    {
        if (charge_name == "plus-half-minus-half")
            return TwoDefectCharge::plus_half_minus_half;
        else if (charge_name == "plus-half-minus-half-alt")
            return TwoDefectCharge::plus_half_minus_half_alt;
        else if (charge_name == "plus-half-plus-half")
            return TwoDefectCharge::plus_half_plus_half;
        else
            throw std::invalid_argument("Inputted incorrect charge name for two defect configuration");
    }

    template <int dim>
    std::vector<dealii::Point<dim>>
    parse_centers_from_vector(const std::vector<std::vector<double>> &centers_vector)
    {
        const int num_defects = 2;
        if (centers_vector.size() != num_defects)
            throw std::invalid_argument("Wrong number of defect centers in "
                                        "parameter file");

        std::vector<dealii::Point<dim>> centers(num_defects, dealii::Point<dim>());

        // centers_vec should be in order of ((x_1, y_1), (x_2, y_2))
        for (unsigned int n = 0; n < num_defects; ++n)
            for (unsigned int i = 0; i < dim; ++i)
                centers[n][i] = centers_vector[n][i];

        return centers;

    }

    template <int dim>
    typename TwoDefectConfiguration<dim>::DefectAxis get_axis_from_name(const std::string &axis_name)
    {
        if (axis_name == "x")
            return TwoDefectConfiguration<dim>::DefectAxis::x;
        else if (axis_name == "y")
            return TwoDefectConfiguration<dim>::DefectAxis::y;
        else if (axis_name == "z")
            return TwoDefectConfiguration<dim>::DefectAxis::z;
        else
            throw std::invalid_argument("Axis name in TwoDefectConfiguration must be x, y, or z");
    }

    template <int dim>
    double calc_phi(const dealii::Tensor<1, dim> &p, const typename TwoDefectConfiguration<dim>::DefectAxis axis);

    template <>
    double calc_phi<3>(const dealii::Tensor<1, 3> &p, const typename TwoDefectConfiguration<3>::DefectAxis axis)
    {
        switch(axis)
        {
        case TwoDefectConfiguration<3>::DefectAxis::x:
	        return std::atan2(p[2], p[1]);
        case TwoDefectConfiguration<3>::DefectAxis::y:
	        return std::atan2(p[0], p[2]);
        case TwoDefectConfiguration<3>::DefectAxis::z:
	        return std::atan2(p[1], p[0]);
        default:
            throw std::invalid_argument("Axis name in TwoDefectConfiguration must be x, y, or z");
        }
    }

    template <>
    double calc_phi<2>(const dealii::Tensor<1, 2> &p, const typename TwoDefectConfiguration<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwoDefectConfiguration<2>::DefectAxis::x:
 	        return std::atan2(0, p[1]);
        case TwoDefectConfiguration<2>::DefectAxis::y:
 	        return std::atan2(p[0], 0);
        case TwoDefectConfiguration<2>::DefectAxis::z:
 	        return std::atan2(p[1], p[0]);
        default:
            throw std::invalid_argument("Axis name in TwoDefectConfiguration must be x, y, or z");
        }
    }


    template <int dim>
    double calc_r(const dealii::Tensor<1, dim> &p, const typename TwoDefectConfiguration<dim>::DefectAxis axis);

    template <>
    double calc_r<3>(const dealii::Tensor<1, 3> &p, const typename TwoDefectConfiguration<3>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwoDefectConfiguration<3>::DefectAxis::x:
 	        return std::sqrt(p[2]*p[2] + p[1]*p[1]);
        case TwoDefectConfiguration<3>::DefectAxis::y:
 	        return std::sqrt(p[2]*p[2] + p[0]*p[0]);
        case TwoDefectConfiguration<3>::DefectAxis::z:
 	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        default:
            throw std::invalid_argument("Axis name in TwoDefectConfiguration must be x, y, or z");
        }
    }
    
    template <>
    double calc_r<2>(const dealii::Tensor<1, 2> &p, const typename TwoDefectConfiguration<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwoDefectConfiguration<2>::DefectAxis::x:
 	        return std::abs(p[1]);
        case TwoDefectConfiguration<2>::DefectAxis::y:
 	        return std::abs(p[0]);
        case TwoDefectConfiguration<2>::DefectAxis::z:
 	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        default:
            throw std::invalid_argument("Axis name in TwoDefectConfiguration must be x, y, or z");
        }
    }

    template <int dim>
    dealii::Point<3> calc_n(double phi, const typename TwoDefectConfiguration<dim>::DefectAxis axis);

    template <>
    dealii::Point<3> calc_n<3>(double theta, const typename TwoDefectConfiguration<3>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwoDefectConfiguration<3>::DefectAxis::x:
 	        return dealii::Point<3>({0, std::cos(theta), std::sin(theta)});
        case TwoDefectConfiguration<3>::DefectAxis::y:
 	        return dealii::Point<3>({std::sin(theta), 0, std::cos(theta)});
        case TwoDefectConfiguration<3>::DefectAxis::z:
 	        return dealii::Point<3>({std::cos(theta), std::sin(theta), 0});
        default:
            throw std::invalid_argument("Axis name in TwoDefectConfiguration must be x, y, or z");
        }
    }

    template <>
    dealii::Point<3> calc_n<2>(double theta, const typename TwoDefectConfiguration<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwoDefectConfiguration<2>::DefectAxis::x:
 	        return dealii::Point<3>({0, std::cos(theta), std::sin(theta)});
        case TwoDefectConfiguration<2>::DefectAxis::y:
 	        return dealii::Point<3>({std::sin(theta), 0, std::cos(theta)});
        case TwoDefectConfiguration<2>::DefectAxis::z:
 	        return dealii::Point<3>({std::cos(theta), std::sin(theta), 0});
        default:
            throw std::invalid_argument("Axis name in TwoDefectConfiguration must be x, y, or z");
        }
    }
}



template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration()
    : BoundaryValues<dim>("plus-half-minus-half")
    , k(return_defect_charge_val(charge))
{}



template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration(double S0_,
                                                    TwoDefectCharge charge_,
                                                    std::vector<std::vector<double>> centers_)
  : BoundaryValues<dim>(return_defect_name(charge_))
  , S0(S0_)
  , charge(charge_)
  , k(return_defect_charge_val(charge_))
  , centers(parse_centers_from_vector<dim>(centers_))
{}




template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration(std::map<std::string, boost::any> &am)
    : BoundaryValues<dim>(boost::any_cast<std::string>(am["defect-charge-name"]),
                          boost::any_cast<std::string>(am["boundary-condition"]))
    , S0(boost::any_cast<double>(am["S-value"]))
    , charge(get_charge_from_name(boost::any_cast<std::string>(am["defect-charge-name"])))
    , k(return_defect_charge_val(charge))
    , centers(parse_centers_from_vector<dim>(
                boost::any_cast<std::vector<std::vector<double>>>(am["defect-positions"])))
    , axis(get_axis_from_name<dim>(boost::any_cast<std::string>(am["defect-axis"])))
{
    for (const auto &center : centers)
        this->defect_pts.push_back(center);
}



/* TODO: Be able to specify center locations from the command line  */
template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration(po::variables_map vm)
  : BoundaryValues<dim>(vm["defect-charge-name"].as<std::string>())
  , S0(vm["S-value"].as<double>())
  , charge(get_charge_from_name(vm["defect-charge-name"].as<std::string>()))
  , k(return_defect_charge_val(charge))
  , centers(parse_centers_from_vector<dim>(
              vm["centers"].as<std::vector<std::vector<double>>>()))
{}



template <int dim>
double TwoDefectConfiguration<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
    const double phi_1 = calc_phi(p - centers[0], axis);
    const double phi_2 = calc_phi(p - centers[1], axis);
    const double theta = charge == TwoDefectCharge::plus_half_minus_half_alt ? 
                         k[0] * phi_1 + k[1] * phi_2 + 0.5 * M_PI
                         : k[0] * phi_1 + k[1] * phi_2;
  
    const double r1 = calc_r(p - centers[0], axis);
    const double r2 = calc_r(p - centers[1], axis);

    const double S1 = 2 * (1.0 / (1 + std::exp(-r1)) - 1.0);
    const double S2 = 2 * (1.0 / (1 + std::exp(-r2)) - 1.0);
    const double S = S0 * (1.0 + (S1 + S2));

    const auto n = calc_n<dim>(theta, axis);

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
        throw std::invalid_argument("In TwoDefectConfiguration::value `component` must be 0 to 4");
	}
}



template <int dim>
void TwoDefectConfiguration<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double>   &value) const
{
    const double phi_1 = calc_phi(p - centers[0], axis);
    const double phi_2 = calc_phi(p - centers[1], axis);
    const double theta = charge == TwoDefectCharge::plus_half_minus_half_alt ? 
                         k[0] * phi_1 + k[1] * phi_2 + 0.5 * M_PI
                         : k[0] * phi_1 + k[1] * phi_2;
  
    const double r1 = calc_r(p - centers[0], axis);
    const double r2 = calc_r(p - centers[1], axis);

    const double S1 = 2 * (1.0 / (1 + std::exp(-r1)) - 1.0);
    const double S2 = 2 * (1.0 / (1 + std::exp(-r2)) - 1.0);
    const double S = S0 * (1.0 + (S1 + S2));

    const auto n = calc_n<dim>(theta, axis);

	value[0] = S * (n[0]*n[0] - 1.0/3.0);
	value[1] = S * n[0]*n[1];
	value[2] = S * n[0]*n[2];
	value[3] = S * (n[1]*n[1] - 1.0/3.0);
	value[4] = S * n[1]*n[2];
}



template <int dim>
void TwoDefectConfiguration<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
    double phi_1 = 0;
    double phi_2 = 0;
    double r1 = 0;
    double r2 = 0;
    double theta = 0;
    double S1 = 0;
    double S2 = 0;
    double S = 0;
    dealii::Point<3> n;

    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        phi_1 = calc_phi(point_list[i] - centers[0], axis);
        phi_2 = calc_phi(point_list[i] - centers[1], axis);
        theta = charge == TwoDefectCharge::plus_half_minus_half_alt ? 
                k[0] * phi_1 + k[1] * phi_2 + 0.5 * M_PI
                : k[0] * phi_1 + k[1] * phi_2;
  
        r1 = calc_r(point_list[i] - centers[0], axis);
        r2 = calc_r(point_list[i] - centers[1], axis);

        S1 = 2 * (1.0 / (1 + std::exp(-r1)) - 1.0);
        S2 = 2 * (1.0 / (1 + std::exp(-r2)) - 1.0);
        S = S0 * (1.0 + (S1 + S2));

        n = calc_n<dim>(theta, axis);

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
        default:
            throw std::invalid_argument("In TwoDefectConfiguration::value_list `component` must be 0 to 4");
        }
    }
}



/** DIMENSIONALLY-WEIRD projects distances + angles into x-y plane */
template <int dim>
void TwoDefectConfiguration<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    double phi_1 = 0;
    double phi_2 = 0;
    double r1 = 0;
    double r2 = 0;
    double theta = 0;
    double S1 = 0;
    double S2 = 0;
    double S = 0;
    dealii::Point<3> n;

    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        phi_1 = calc_phi(point_list[i] - centers[0], axis);
        phi_2 = calc_phi(point_list[i] - centers[1], axis);
        theta = charge == TwoDefectCharge::plus_half_minus_half_alt ? 
                k[0] * phi_1 + k[1] * phi_2 + 0.5 * M_PI
                : k[0] * phi_1 + k[1] * phi_2;
  
        r1 = calc_r(point_list[i] - centers[0], axis);
        r2 = calc_r(point_list[i] - centers[1], axis);

        S1 = 2 * (1.0 / (1 + std::exp(-r1)) - 1.0);
        S2 = 2 * (1.0 / (1 + std::exp(-r2)) - 1.0);
        S = S0 * (1.0 + (S1 + S2));

        n = calc_n<dim>(theta, axis);

        value_list[i][0] = S * (n[0]*n[0] - 1.0/3.0);
        value_list[i][1] = S * n[0]*n[1];
        value_list[i][2] = S * n[0]*n[2];
        value_list[i][3] = S * (n[1]*n[1] - 1.0/3.0);
        value_list[i][4] = S * n[1]*n[2];
    }
}

template class TwoDefectConfiguration<3>;
template class TwoDefectConfiguration<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(TwoDefectConfiguration<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(TwoDefectConfiguration<3>)
