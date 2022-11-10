#include "PeriodicConfiguration.hpp"

#include <boost/any.hpp>
#include <boost/serialization/export.hpp>

#include <map>
#include <string>

template <int dim>
PeriodicConfiguration<dim>::PeriodicConfiguration(double k_,
                                                  double eps_, double S_)
    : k(k_)
    , eps(eps_)
    , S(S_)
    , BoundaryValues<dim>(std::string("periodic"))
{}



template <int dim>
PeriodicConfiguration<dim>::
PeriodicConfiguration(std::map<std::string, boost::any> &am)
    : k(boost::any_cast<double>(am["k"]))
    , eps(boost::any_cast<double>(am["eps"]))
    , S(boost::any_cast<double>(am["S-value"]))
    , BoundaryValues<dim>(std::string("periodic"),
                          boost::any_cast<std::string>(am["boundary-condition"]))
{}



template <int dim>
double PeriodicConfiguration<dim>::
value(const dealii::Point<dim> &p, const unsigned int component) const
{
    switch (component)
    {
    case 0:
        return S * (2.0 / 3.0);
        break;
    case 1:
        return S * eps * std::sin(k * p[0]);
        break;
    case 2:
        return 0;
        break;
    case 3:
        return S * (-1.0 / 3.0);
        break;
    case 4:
        return 0;
        break;
    default:
        throw std::invalid_argument("Invalid component value");
    }
}



template <int dim>
void PeriodicConfiguration<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double> &value) const
{
    value[0] = S * (2.0 / 3.0);
    value[1] = S * eps * std::sin(k * p[0]);
    value[2] = 0;
    value[3] = S * (-1.0 / 3.0);
    value[4] = 0;
}



template <int dim>
void PeriodicConfiguration<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double> &value_list,
           const unsigned int component) const
{
    switch (component)
    {
    case 0:
        for (std::size_t i = 0; i < point_list.size(); ++i)
            value_list[i] = S * (2.0 / 3.0);
        break;
    case 1:
        for (std::size_t i = 0; i < point_list.size(); ++i)
            value_list[i] = S * eps * std::sin(k * point_list[i][0]);
        break;
    case 2:
        for (std::size_t i = 0; i < point_list.size(); ++i)
            value_list[i] = 0;
        break;
    case 3:
        for (std::size_t i = 0; i < point_list.size(); ++i)
            value_list[i] =  S * (-1.0 / 3.0);
        break;
    case 4:
      for (std::size_t i = 0; i < point_list.size(); ++i)
        value_list[i] = 0;
      break;
    }
}



template <int dim>
void PeriodicConfiguration<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>> &value_list) const
{
    for (std::size_t i = 0; i < point_list.size(); ++i)
    {
        value_list[i][0] = S * (2.0 / 3.0);
        value_list[i][1] = S * eps * std::sin(k * point_list[i][0]);
        value_list[i][2] = 0;
        value_list[i][3] = S * (-1.0 / 3.0);
        value_list[i][4] = 0;
    }
}

template class PeriodicConfiguration<3>;
template class PeriodicConfiguration<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(PeriodicConfiguration<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(PeriodicConfiguration<3>)
