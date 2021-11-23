#include "DefectConfiguration.hpp"
#include "maier_saupe_constants.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

namespace msc = maier_saupe_constants;

template <int dim>
DefectConfiguration<dim>::DefectConfiguration(double S_, double k_)
    : dealii::Function<dim>(msc::vec_dim<dim>)
	, S(S_)
    , k(k_)
{}



template <int dim>
double DefectConfiguration<dim>::value
(const dealii::Point<dim> &p, const unsigned int component) const
{
	double phi = std::atan2(p[1], p[0]);
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
	switch (component)
	{
	case 0:
        for (int i = 0; i < point_list.size(); ++i)
		{
			phi = std::atan2(point_list[i][1], point_list[i][0]);
		    value_list[i] = 0.5 * S * ( 1.0/3.0 + std::cos(2*k*phi) );
		}
		break;
	case 1:
        for (int i = 0; i < point_list.size(); ++i)
		{
			phi = std::atan2(point_list[i][1], point_list[i][0]);
		    value_list[i] = 0.5 * S * std::sin(2*k*phi);
		}
		break;
	case 2:
        for (int i = 0; i < point_list.size(); ++i)
		    value_list[i] = 0.0;
		break;
	case 3:
        for (int i = 0; i < point_list.size(); ++i)
		{
			phi = std::atan2(point_list[i][1], point_list[i][0]);
		    value_list[i] = 0.5 * S * ( 1.0/3.0 - std::cos(2*k*phi) );
		}
		break;
	case 4:
        for (int i = 0; i < point_list.size(); ++i)
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
    for (int i = 0; i < point_list.size(); ++i)
    { 
		phi = std::atan2(point_list[i][1], point_list[i][0]);

	    value_list[i][0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*k*phi) );
	    value_list[i][1] = 0.5 * S * std::sin(2*k*phi);
	    value_list[i][2] = 0.0;
	    value_list[i][3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*k*phi) );
	    value_list[i][4] = 0.0;
    }
}

template class DefectConfiguration<3>;
template class DefectConfiguration<2>;