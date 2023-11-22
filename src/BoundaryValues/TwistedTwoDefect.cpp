#include "TwistedTwoDefect.hpp"
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
    typename TwistedTwoDefect<dim>::DefectAxis get_axis_from_name(const std::string &axis_name)
    {
        if (axis_name == "x")
            return TwistedTwoDefect<dim>::DefectAxis::x;
        else if (axis_name == "y")
            return TwistedTwoDefect<dim>::DefectAxis::y;
        else if (axis_name == "z")
            return TwistedTwoDefect<dim>::DefectAxis::z;
        else
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
    }

    template <int dim>
    double calc_phi(const dealii::Tensor<1, dim> &p, const typename TwistedTwoDefect<dim>::DefectAxis axis);

    template <>
    double calc_phi<3>(const dealii::Tensor<1, 3> &p, const typename TwistedTwoDefect<3>::DefectAxis axis)
    {
        switch(axis)
        {
        case TwistedTwoDefect<3>::DefectAxis::x:
	        return std::atan2(p[2], p[1]);
        case TwistedTwoDefect<3>::DefectAxis::y:
	        return std::atan2(p[0], p[2]);
        case TwistedTwoDefect<3>::DefectAxis::z:
	        return std::atan2(p[1], p[0]);
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    template <>
    double calc_phi<2>(const dealii::Tensor<1, 2> &p, const typename TwistedTwoDefect<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<2>::DefectAxis::x:
 	        return std::atan2(0, p[1]);
        case TwistedTwoDefect<2>::DefectAxis::y:
 	        return std::atan2(p[0], 0);
        case TwistedTwoDefect<2>::DefectAxis::z:
 	        return std::atan2(p[1], p[0]);
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }


    template <int dim>
    double calc_r(const dealii::Tensor<1, dim> &p, const typename TwistedTwoDefect<dim>::DefectAxis axis);

    template <>
    double calc_r<3>(const dealii::Tensor<1, 3> &p, const typename TwistedTwoDefect<3>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<3>::DefectAxis::x:
 	        return std::sqrt(p[2]*p[2] + p[1]*p[1]);
        case TwistedTwoDefect<3>::DefectAxis::y:
 	        return std::sqrt(p[2]*p[2] + p[0]*p[0]);
        case TwistedTwoDefect<3>::DefectAxis::z:
 	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }
    
    template <>
    double calc_r<2>(const dealii::Tensor<1, 2> &p, const typename TwistedTwoDefect<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<2>::DefectAxis::x:
 	        return std::abs(p[1]);
        case TwistedTwoDefect<2>::DefectAxis::y:
 	        return std::abs(p[0]);
        case TwistedTwoDefect<2>::DefectAxis::z:
 	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    template <int dim>
    dealii::Point<3> calc_n(double phi, const typename TwistedTwoDefect<dim>::DefectAxis axis);

    template <>
    dealii::Point<3> calc_n<3>(double theta, const typename TwistedTwoDefect<3>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<3>::DefectAxis::x:
 	        return dealii::Point<3>({0, std::cos(theta), std::sin(theta)});
        case TwistedTwoDefect<3>::DefectAxis::y:
 	        return dealii::Point<3>({std::sin(theta), 0, std::cos(theta)});
        case TwistedTwoDefect<3>::DefectAxis::z:
 	        return dealii::Point<3>({std::cos(theta), std::sin(theta), 0});
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    template <>
    dealii::Point<3> calc_n<2>(double theta, const typename TwistedTwoDefect<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<2>::DefectAxis::x:
 	        return dealii::Point<3>({0, std::cos(theta), std::sin(theta)});
        case TwistedTwoDefect<2>::DefectAxis::y:
 	        return dealii::Point<3>({std::sin(theta), 0, std::cos(theta)});
        case TwistedTwoDefect<2>::DefectAxis::z:
 	        return dealii::Point<3>({std::cos(theta), std::sin(theta), 0});
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    template <int dim>
    dealii::Point<3> calc_m(double phi, const typename TwistedTwoDefect<dim>::DefectAxis axis);

    template <>
    dealii::Point<3> calc_m<3>(double theta, const typename TwistedTwoDefect<3>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<3>::DefectAxis::x:
 	        return dealii::Point<3>({0, -std::sin(theta), std::cos(theta)});
        case TwistedTwoDefect<3>::DefectAxis::y:
 	        return dealii::Point<3>({std::cos(theta), 0, -std::sin(theta)});
        case TwistedTwoDefect<3>::DefectAxis::z:
 	        return dealii::Point<3>({-std::sin(theta), std::cos(theta), 0});
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    template <>
    dealii::Point<3> calc_m<2>(double theta, const typename TwistedTwoDefect<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<2>::DefectAxis::x:
 	        return dealii::Point<3>({0, -std::sin(theta), std::cos(theta)});
        case TwistedTwoDefect<2>::DefectAxis::y:
 	        return dealii::Point<3>({std::cos(theta), 0, -std::sin(theta)});
        case TwistedTwoDefect<2>::DefectAxis::z:
 	        return dealii::Point<3>({-std::sin(theta), std::cos(theta), 0});
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    template <int dim>
    dealii::Point<3> calc_l(double phi, const typename TwistedTwoDefect<dim>::DefectAxis axis);

    template <>
    dealii::Point<3> calc_l<3>(double theta, const typename TwistedTwoDefect<3>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<3>::DefectAxis::x:
 	        return dealii::Point<3>({1, 0, 0});
        case TwistedTwoDefect<3>::DefectAxis::y:
 	        return dealii::Point<3>({0, 1, 0});
        case TwistedTwoDefect<3>::DefectAxis::z:
 	        return dealii::Point<3>({0, 0, 1});
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    template <>
    dealii::Point<3> calc_l<2>(double theta, const typename TwistedTwoDefect<2>::DefectAxis axis)
    {
        switch (axis)
        {
        case TwistedTwoDefect<2>::DefectAxis::x:
 	        return dealii::Point<3>({1, 0, 0});
        case TwistedTwoDefect<2>::DefectAxis::y:
 	        return dealii::Point<3>({0, 1, 0});
        case TwistedTwoDefect<2>::DefectAxis::z:
 	        return dealii::Point<3>({0, 0, 1});
        default:
            throw std::invalid_argument("Axis name in TwistedTwoDefect must be x, y, or z");
        }
    }

    double calc_q1(double r1, double r2, double qmax, double qmin)
    {
        return (qmax - qmin) * (-1 + std::tanh(r1) + std::tanh(r2)) + qmin;
    }

    double calc_q2(double r1, double r2, double qmax, double qmin)
    {
        return 0;
        // return (qmax - qmin) * (2 - std::tanh(r1) - std::tanh(r2)) + qmin;
    }
}



template <int dim>
TwistedTwoDefect<dim>::TwistedTwoDefect()
    : BoundaryValues<dim>("plus-half-minus-half")
    , k(return_defect_charge_val(charge))
{}



template <int dim>
TwistedTwoDefect<dim>::TwistedTwoDefect(std::map<std::string, boost::any> &am)
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



template <int dim>
double TwistedTwoDefect<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
    const double phi_1 = calc_phi(p - centers[0], axis);
    const double phi_2 = calc_phi(p - centers[1], axis);
    const double theta = charge == TwoDefectCharge::plus_half_minus_half_alt ? 
                         k[0] * phi_1 + k[1] * phi_2 + 0.5 * M_PI
                         : k[0] * phi_1 + k[1] * phi_2;
  
    const double r1 = calc_r(p - centers[0], axis);
    const double r2 = calc_r(p - centers[1], axis);

    const double q1max = (2.0 / 3.0) * S0;
    const double q1min = 0.25 * q1max;

    const double q2max = q1min;
    const double q2min = -0.5 * q1max;

    const double q1 = calc_q1(r1, r2, q1max, q1min);
    const double q2 = calc_q2(r1, r2, q2max, q2min);

    const auto n = calc_n<dim>(theta, axis);
    const auto m = calc_m<dim>(theta, axis);
    const auto l = calc_l<dim>(theta, axis);

    switch (component)
    {
	case 0:
        return q1 * n[0]*n[0] + q2 * m[0]*m[0] - (q1 + q2) * l[0]*l[0];
	case 1:
        return q1 * n[0]*n[1] + q2 * m[0]*m[1] - (q1 + q2) * l[0]*l[1];
	case 2:
        return q1 * n[0]*n[2] + q2 * m[0]*m[2] - (q1 + q2) * l[0]*l[2];
	case 3:
        return q1 * n[1]*n[1] + q2 * m[1]*m[1] - (q1 + q2) * l[1]*l[1];
	case 4:
        return q1 * n[1]*n[2] + q2 * m[1]*m[2] - (q1 + q2) * l[1]*l[2];
    default:
        throw std::invalid_argument("In TwistedTwoDefect::value `component` must be 0 to 4");
	}
}



template <int dim>
void TwistedTwoDefect<dim>::
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

    const double q1max = (2.0 / 3.0) * S0;
    const double q1min = 0.25 * q1max;

    const double q2max = q1min;
    const double q2min = -0.5 * q1max;

    const double q1 = calc_q1(r1, r2, q1max, q1min);
    const double q2 = calc_q2(r1, r2, q2max, q2min);

    const auto n = calc_n<dim>(theta, axis);
    const auto m = calc_m<dim>(theta, axis);
    const auto l = calc_l<dim>(theta, axis);

	value[0] = q1 * n[0]*n[0] + q2 * m[0]*m[0] - (q1 + q2) * l[0]*l[0];
	value[1] = q1 * n[0]*n[1] + q2 * m[0]*m[1] - (q1 + q2) * l[0]*l[1];
	value[2] = q1 * n[0]*n[2] + q2 * m[0]*m[2] - (q1 + q2) * l[0]*l[2];
	value[3] = q1 * n[1]*n[1] + q2 * m[1]*m[1] - (q1 + q2) * l[1]*l[1];
	value[4] = q1 * n[1]*n[2] + q2 * m[1]*m[2] - (q1 + q2) * l[1]*l[2];
}



template <int dim>
void TwistedTwoDefect<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
    const double q1max = (2.0 / 3.0) * S0;
    const double q1min = 0.25 * q1max;

    const double q2max = q1min;
    const double q2min = -0.5 * q1max;

    double phi_1 = 0;
    double phi_2 = 0;
    double r1 = 0;
    double r2 = 0;
    double theta = 0;
    double q1 = 0;
    double q2 = 0;
    dealii::Point<3> n;
    dealii::Point<3> m;
    dealii::Point<3> l;

    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        phi_1 = calc_phi(point_list[i] - centers[0], axis);
        phi_2 = calc_phi(point_list[i] - centers[1], axis);
        theta = charge == TwoDefectCharge::plus_half_minus_half_alt ? 
                k[0] * phi_1 + k[1] * phi_2 + 0.5 * M_PI
                : k[0] * phi_1 + k[1] * phi_2;
  
        r1 = calc_r(point_list[i] - centers[0], axis);
        r2 = calc_r(point_list[i] - centers[1], axis);

        q1 = calc_q1(r1, r2, q1max, q1min);
        q2 = calc_q2(r1, r2, q2max, q2min);

        n = calc_n<dim>(theta, axis);
        m = calc_m<dim>(theta, axis);
        l = calc_l<dim>(theta, axis);

        switch (component)
        {
	    case 0:
            value_list[i] = q1 * n[0]*n[0] + q2 * m[0]*m[0] - (q1 + q2) * l[0]*l[0];
            break;
	    case 1:
            value_list[i] = q1 * n[0]*n[1] + q2 * m[0]*m[1] - (q1 + q2) * l[0]*l[1];
            break;
	    case 2:
            value_list[i] = q1 * n[0]*n[2] + q2 * m[0]*m[2] - (q1 + q2) * l[0]*l[2];
            break;
	    case 3:
            value_list[i] = q1 * n[1]*n[1] + q2 * m[1]*m[1] - (q1 + q2) * l[1]*l[1];
            break;
	    case 4:
            value_list[i] = q1 * n[1]*n[2] + q2 * m[1]*m[2] - (q1 + q2) * l[1]*l[2];
            break;
        default:
            throw std::invalid_argument("In TwistedTwoDefect::value `component` must be 0 to 4");
        }
    }
}



/** DIMENSIONALLY-WEIRD projects distances + angles into x-y plane */
template <int dim>
void TwistedTwoDefect<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    const double q1max = (2.0 / 3.0) * S0;
    const double q1min = 0.25 * q1max;

    const double q2max = q1min;
    const double q2min = -0.5 * q1max;

    double phi_1 = 0;
    double phi_2 = 0;
    double r1 = 0;
    double r2 = 0;
    double theta = 0;
    double q1 = 0;
    double q2 = 0;
    dealii::Point<3> n;
    dealii::Point<3> m;
    dealii::Point<3> l;

    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        phi_1 = calc_phi(point_list[i] - centers[0], axis);
        phi_2 = calc_phi(point_list[i] - centers[1], axis);
        theta = charge == TwoDefectCharge::plus_half_minus_half_alt ? 
                k[0] * phi_1 + k[1] * phi_2 + 0.5 * M_PI
                : k[0] * phi_1 + k[1] * phi_2;
  
        r1 = calc_r(point_list[i] - centers[0], axis);
        r2 = calc_r(point_list[i] - centers[1], axis);

        q1 = calc_q1(r1, r2, q1max, q1min);
        q2 = calc_q2(r1, r2, q2max, q2min);

        n = calc_n<dim>(theta, axis);
        m = calc_m<dim>(theta, axis);
        l = calc_l<dim>(theta, axis);

	    // value_list[i][0] = q1 * (n[0]*n[0] - 1.0/3.0); // + q2 * m[0]*m[0] - (q1 + q2) * l[0]*l[0];
	    // value_list[i][1] = q1 * n[0]*n[1]; // + q2 * m[0]*m[1] - (q1 + q2) * l[0]*l[1];
	    // value_list[i][2] = q1 * n[0]*n[2]; // + q2 * m[0]*m[2] - (q1 + q2) * l[0]*l[2];
	    // value_list[i][3] = q1 * (n[1]*n[1] - 1.0/3.0); // + q2 * m[1]*m[1] - (q1 + q2) * l[1]*l[1];
	    // value_list[i][4] = q1 * n[1]*n[2]; // + q2 * m[1]*m[2] - (q1 + q2) * l[1]*l[2];

	    value_list[i][0] = q1;
	    value_list[i][1] = 0;
	    value_list[i][2] = 0;
	    value_list[i][3] = 0;
	    value_list[i][4] = 0;
    }
}

template class TwistedTwoDefect<3>;
template class TwistedTwoDefect<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(TwistedTwoDefect<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(TwistedTwoDefect<3>)
