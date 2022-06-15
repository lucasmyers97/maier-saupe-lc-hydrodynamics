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
            default:
                throw std::invalid_argument("Inputted incorrect charge for two defect configuration");
            }
    }

    TwoDefectCharge get_charge_from_name(const std::string charge_name)
    {
        if (charge_name == "plus-half-minus-half")
        {
            return TwoDefectCharge::plus_half_minus_half;
        } else if (charge_name == "plus-half-minus-half-alt")
        {
            return TwoDefectCharge::plus_half_minus_half_alt;
        } else
        {
            throw std::invalid_argument("Inputted incorrect charge name for two defect configuration");
        }
    }

    template <int dim>
    std::vector<dealii::Point<dim>>
    parse_centers_from_vector(const std::vector<double> centers_vector)
    {
        const int num_defects = 2;
        std::vector<dealii::Point<dim>> centers(dim, dealii::Point<dim>());

        // centers_vec should be in order of ((x_1, y_1), (x_2, y_2))
        for (unsigned int n = 0; n < num_defects; ++n)
            for (unsigned int i = 0; i < dim; ++i)
                centers[n][i] = centers_vector[n*num_defects + i];

        return centers;

    }
}



template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration()
    : BoundaryValues<dim>("plus-half-minus-half")
    , k(return_defect_charge_val(charge))
{}



template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration(double S_,
                                                    TwoDefectCharge charge_,
                                                    std::vector<double> centers_)
  : S(S_)
  , charge(charge_)
  , BoundaryValues<dim>(return_defect_name(charge_))
  , k(return_defect_charge_val(charge_))
  , centers(parse_centers_from_vector<dim>(centers_))
{}




template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration(std::map<std::string, boost::any> &am)
    : S(boost::any_cast<double>(am["S-value"]))
    , charge(get_charge_from_name(boost::any_cast<std::string>(am["defect-charge-name"])))
    , BoundaryValues<dim>(boost::any_cast<std::string>(am["defect-charge-name"]))
    , k(return_defect_charge_val(charge))
    , centers(parse_centers_from_vector<dim>(boost::any_cast<std::vector<double>>(am["centers"])))
{}



/* TODO: Be able to specify center locations from the command line  */
template <int dim>
TwoDefectConfiguration<dim>::TwoDefectConfiguration(po::variables_map vm)
  : S(vm["S-value"].as<double>())
  , charge(get_charge_from_name(vm["defect-charge-name"].as<std::string>()))
  , BoundaryValues<dim>(vm["defect-charge-name"].as<std::string>())
  , k(return_defect_charge_val(charge))
  , centers(parse_centers_from_vector<dim>(vm["centers"].as<std::vector<double>>()))
{}


/* TODO: Rotate phi if we have the alternate charge configuration  */
template <int dim>
double TwoDefectConfiguration<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
    double phi_1 = k[0] * std::atan2(p[1] - centers[0][1], p[0] - centers[0][0]);
    double phi_2 = k[1] * std::atan2(p[1] - centers[1][1], p[0] - centers[1][0]);
    double phi = phi_1 + phi_2;
    double return_value = 0;
    if (charge == TwoDefectCharge::plus_half_minus_half_alt)
        phi += 0.5 * M_PI;

    switch (component)
    {
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
            return_value = 0.0;
            break;
    }

    return return_value;
}



template <int dim>
void TwoDefectConfiguration<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double>   &value) const
{
    double phi_1 = k[0] * std::atan2(p[1] - centers[0][1], p[0] - centers[0][0]);
    double phi_2 = k[1] * std::atan2(p[1] - centers[1][1], p[0] - centers[1][0]);
    double phi = phi_1 + phi_2;
    if (charge == TwoDefectCharge::plus_half_minus_half_alt)
        phi += 0.5 * M_PI;

    value[0] = 0.5 * S * (1.0 / 3.0 + std::cos(2 * phi));
    value[1] = 0.5 * S * std::sin(2 * phi);
    value[2] = 0.0;
    value[3] = 0.5 * S * (1.0 / 3.0 - std::cos(2 * phi));
    value[4] = 0.0;
}



template <int dim>
void TwoDefectConfiguration<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double>                   &value_list,
           const unsigned int                    component) const
{
    double phi_1 = 0;
    double phi_2 = 0;
    double phi = 0;
    switch (component)
        {
        case 0:
            for (int i = 0; i < point_list.size(); ++i)
            {
                phi_1 = k[0] * std::atan2(point_list[i][1] - centers[0][1],
                                          point_list[i][0] - centers[0][0]);
                phi_2 = k[1] * std::atan2(point_list[i][1] - centers[1][1],
                                          point_list[i][0] - centers[1][0]);
                phi = phi_1 + phi_2;
                if (charge == TwoDefectCharge::plus_half_minus_half_alt)
                    phi += 0.5 * M_PI;
                value_list[i] = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
            }
            break;
        case 1:
            for (int i = 0; i < point_list.size(); ++i)
            {
                phi_1 = k[0] * std::atan2(point_list[i][1] - centers[0][1],
                                          point_list[i][0] - centers[0][0]);
                phi_2 = k[1] * std::atan2(point_list[i][1] - centers[1][1],
                                          point_list[i][0] - centers[1][0]);
                phi = phi_1 + phi_2;
                if (charge == TwoDefectCharge::plus_half_minus_half_alt)
                    phi += 0.5 * M_PI;
                value_list[i] = 0.5 * S * std::sin(2*phi);
            }
            break;
        case 2:
            for (int i = 0; i < point_list.size(); ++i)
                value_list[i] = 0.0;
            break;
        case 3:
            for (int i = 0; i < point_list.size(); ++i)
            {
                phi_1 = k[0] * std::atan2(point_list[i][1] - centers[0][1],
                                          point_list[i][0] - centers[0][0]);
                phi_2 = k[1] * std::atan2(point_list[i][1] - centers[1][1],
                                          point_list[i][0] - centers[1][0]);
                phi = phi_1 + phi_2;
                if (charge == TwoDefectCharge::plus_half_minus_half_alt)
                    phi += 0.5 * M_PI;
                value_list[i] = 0.5 * S * (1.0 / 3.0 - std::cos(2*phi));
            }
            break;
        case 4:
            for (int i = 0; i < point_list.size(); ++i)
                value_list[i] = 0.0;
            break;
        }
}



template <int dim>
void TwoDefectConfiguration<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    double phi_1 = 0;
    double phi_2 = 0;
    double phi = 0;
    for (int i = 0; i < point_list.size(); ++i)
    {
        phi_1 = k[0] * std::atan2(point_list[i][1] - centers[0][1],
                                  point_list[i][0] - centers[0][0]);
        phi_2 = k[1] * std::atan2(point_list[i][1] - centers[1][1],
                                  point_list[i][0] - centers[1][0]);
        phi = phi_1 + phi_2;
        if (charge == TwoDefectCharge::plus_half_minus_half_alt)
            phi += 0.5 * M_PI;

        value_list[i][0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
        value_list[i][1] = 0.5 * S * std::sin(2*phi);
        value_list[i][2] = 0.0;
        value_list[i][3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*phi) );
        value_list[i][4] = 0.0;
    }
}

template class TwoDefectConfiguration<3>;
template class TwoDefectConfiguration<2>;

