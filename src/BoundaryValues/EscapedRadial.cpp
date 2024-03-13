#include "EscapedRadial.hpp"
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

namespace
{
    template <int dim>
    dealii::Point<dim>
    parse_center_from_vector(const std::vector<double> &center_vector)
    {
        if (center_vector.size() != dim)
            throw std::invalid_argument("Wrong number of ER axis center "
                                        "coordinates in parameter file");

        dealii::Point<dim> center;

        for (unsigned int i = 0; i < dim; ++i)
            center[i] = center_vector[i];

        return center;
    }

    template <int dim>
    typename EscapedRadial<dim>::Axis get_axis_from_name(const std::string &axis_name)
    {
        if (axis_name == "x")
            return EscapedRadial<dim>::Axis::x;
        else if (axis_name == "y")
            return EscapedRadial<dim>::Axis::y;
        else if (axis_name == "z")
            return EscapedRadial<dim>::Axis::z;
        else
            throw std::invalid_argument("Axis name in EscapedRadial must be x, y, or z");
    }

    template <int dim>
    double calc_phi(const dealii::Tensor<1, dim> &p, const typename EscapedRadial<dim>::Axis axis);

    template <>
    double calc_phi<2>(const dealii::Tensor<1, 2> &p, const typename EscapedRadial<2>::Axis axis)
    {
        switch(axis)
        {
        case EscapedRadial<2>::Axis::x:
	        return std::atan2(0, p[1]);
        case EscapedRadial<2>::Axis::y:
	        return std::atan2(p[0], 0);
        case EscapedRadial<2>::Axis::z:
	        return std::atan2(p[1], p[0]);
        default:
            throw std::invalid_argument("Axis name in EscapedRadial must be x, y, or z");
        }
    }

    template <>
    double calc_phi<3>(const dealii::Tensor<1, 3> &p, const typename EscapedRadial<3>::Axis axis)
    {
        switch(axis)
        {
        case EscapedRadial<3>::Axis::x:
	        return std::atan2(p[2], p[1]);
        case EscapedRadial<3>::Axis::y:
	        return std::atan2(p[0], p[2]);
        case EscapedRadial<3>::Axis::z:
	        return std::atan2(p[1], p[0]);
        default:
            throw std::invalid_argument("Axis name in EscapedRadial must be x, y, or z");
        }
    }

    template <int dim>
    double calc_r(const dealii::Tensor<1, dim> &p, const typename EscapedRadial<dim>::Axis axis);

    template <>
    double calc_r<2>(const dealii::Tensor<1, 2> &p, const typename EscapedRadial<2>::Axis axis)
    {
        switch (axis)
        {
        case EscapedRadial<2>::Axis::x:
 	        return std::sqrt(p[1]*p[1]);
        case EscapedRadial<2>::Axis::y:
 	        return std::sqrt(p[0]*p[0]);
        case EscapedRadial<2>::Axis::z:
 	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        default:
            throw std::invalid_argument("Axis name in EscapedRadial must be x, y, or z");
        }
    }

    template <>
    double calc_r<3>(const dealii::Tensor<1, 3> &p, const typename EscapedRadial<3>::Axis axis)
    {
        switch (axis)
        {
        case EscapedRadial<3>::Axis::x:
 	        return std::sqrt(p[2]*p[2] + p[1]*p[1]);
        case EscapedRadial<3>::Axis::y:
 	        return std::sqrt(p[2]*p[2] + p[0]*p[0]);
        case EscapedRadial<3>::Axis::z:
 	        return std::sqrt(p[0]*p[0] + p[1]*p[1]);
        default:
            throw std::invalid_argument("Axis name in EscapedRadial must be x, y, or z");
        }
    }

    template <int dim>
    dealii::Tensor<1, dim> calc_n(double phi, double beta, const typename EscapedRadial<dim>::Axis axis);

    template <>
    dealii::Tensor<1, 2> calc_n<2>(double phi, double beta, const typename EscapedRadial<2>::Axis axis)
    {
        switch (axis)
        {
        case EscapedRadial<2>::Axis::x:
 	        return dealii::Tensor<1, 2>({std::cos(beta),
                                         std::cos(phi)*std::sin(beta)});
        case EscapedRadial<2>::Axis::y:
 	        return dealii::Tensor<1, 2>({std::sin(phi)*std::sin(beta),
                                         std::cos(beta)});
        case EscapedRadial<2>::Axis::z:
 	        return dealii::Tensor<1, 2>({std::cos(phi)*std::sin(beta), 
                                         std::sin(phi)*std::sin(beta)});
        default:
            throw std::invalid_argument("Axis name in EscapedRadial must be x, y, or z");
        }
    }

    template <>
    dealii::Tensor<1, 3> calc_n<3>(double phi, double beta, const typename EscapedRadial<3>::Axis axis)
    {
        switch (axis)
        {
        case EscapedRadial<3>::Axis::x:
 	        return dealii::Tensor<1, 3>({std::cos(beta),
                                         std::cos(phi)*std::sin(beta), 
                                         std::sin(phi)*std::sin(beta)});
        case EscapedRadial<3>::Axis::y:
 	        return dealii::Tensor<1, 3>({std::sin(phi)*std::sin(beta),
                                         std::cos(beta),
                                         std::cos(phi)*std::sin(beta)});
        case EscapedRadial<3>::Axis::z:
 	        return dealii::Tensor<1, 3>({std::cos(phi)*std::sin(beta), 
                                         std::sin(phi)*std::sin(beta), 
                                         std::cos(beta)});
        default:
            throw std::invalid_argument("Axis name in EscapedRadial must be x, y, or z");
        }
    }
}



template <int dim>
EscapedRadial<dim>::EscapedRadial()
    : BoundaryValues<dim>("escaped-radial")
{}



template <int dim>
EscapedRadial<dim>::EscapedRadial(std::map<std::string, boost::any> &am)
    : BoundaryValues<dim>("escaped-radial",
                          boost::any_cast<std::string>(am["boundary-condition"]))
    , S0(boost::any_cast<double>(am["S-value"]))
    , cylinder_radius(boost::any_cast<double>(am["cylinder-radius"]))
    , center_axis(
        parse_center_from_vector<dim>(boost::any_cast<std::vector<double>>(am["center-axis"]))
        )
    , axis(get_axis_from_name<dim>(boost::any_cast<std::string>(am["defect-axis"])))
{}



template <int dim>
double EscapedRadial<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
    const double phi = calc_phi(p - center_axis, axis);
    const double r = calc_r(p - center_axis, axis);
    const double beta = 2 * std::atan(r / cylinder_radius);

    const auto n = calc_n<dim>(phi, beta, axis);

    switch (component)
    {
	case 0:
        return S0 * (n[0]*n[0] - 1.0/3.0);
	case 1:
        return S0 * n[0]*n[1];
	case 2:
        return dim == 3 ? S0 * n[0]*n[2] : 0;
	case 3:
        return S0 * (n[1]*n[1] - 1.0/3.0);
	case 4:
        return dim == 3 ? S0 * n[1]*n[2] : 0;
    default:
        throw std::invalid_argument("In EscapedRadial::value `component` must be 0 to 4");
	}
}



template <int dim>
void EscapedRadial<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double>   &value) const
{
    const double phi = calc_phi(p - center_axis, axis);
    const double r = calc_r(p - center_axis, axis);
    const double beta = 2 * std::atan(r / cylinder_radius);

    const auto n = calc_n<dim>(phi, beta, axis);

	value[0] =  S0 * (n[0]*n[0] - 1.0/3.0);
	value[1] =  S0 * n[0]*n[1];
	value[2] =  dim == 3 ? S0 * n[0]*n[2] : 0;
	value[3] =  S0 * (n[1]*n[1] - 1.0/3.0);
	value[4] =  dim == 3 ? S0 * n[1]*n[2] : 0;
}



template <int dim>
void EscapedRadial<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
    double phi = 0;
    double r = 0;
    double beta = 0;
    dealii::Tensor<1, dim> n;

    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        phi = calc_phi(point_list[i] - center_axis, axis);
        r = calc_r(point_list[i] - center_axis, axis);
        beta = 2 * std::atan(r / cylinder_radius);

        n = calc_n<dim>(phi, beta, axis);

        switch (component)
        {
	    case 0:
            value_list[i] = S0 * (n[0]*n[0] - 1.0/3.0);
            break;
	    case 1:
            value_list[i] = S0 * n[0]*n[1];
            break;
	    case 2:
            value_list[i] = dim == 3 ? S0 * n[0]*n[2] : 0;
            break;
	    case 3:
            value_list[i] = S0 * (n[1]*n[1] - 1.0/3.0);
            break;
	    case 4:
            value_list[i] = dim == 3 ? S0 * n[1]*n[2] : 0;
            break;
        default:
            throw std::invalid_argument("In EscapedRadial::value `component` must be 0 to 4");
	    }
    }
}



/** DIMENSIONALLY-WEIRD projects distances + angles into x-y plane */
template <int dim>
void EscapedRadial<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    double phi = 0;
    double r = 0;
    double beta = 0;
    dealii::Tensor<1, dim> n;

    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        phi = calc_phi(point_list[i] - center_axis, axis);
        r = calc_r(point_list[i] - center_axis, axis);
        beta = 2 * std::atan(r / cylinder_radius);

        n = calc_n<dim>(phi, beta, axis);

	    value_list[i][0] =  S0 * (n[0]*n[0] - 1.0/3.0);
	    value_list[i][1] =  S0 * n[0]*n[1];
	    value_list[i][2] =  dim == 3 ? S0 * n[0]*n[2] : 0;
	    value_list[i][3] =  S0 * (n[1]*n[1] - 1.0/3.0);
	    value_list[i][4] =  dim == 3 ? S0 * n[1]*n[2] : 0;
    }
}

template class EscapedRadial<3>;
template class EscapedRadial<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(EscapedRadial<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(EscapedRadial<3>)
