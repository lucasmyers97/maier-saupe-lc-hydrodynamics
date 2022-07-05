#ifndef PERIODIC_CONFIGURATION_HPP
#define PERIODIC_CONFIGURATION_HPP

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

#include <boost/any.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <cmath>
#include <map>
#include <string>

#include "Utilities/maier_saupe_constants.hpp"
#include "BoundaryValues.hpp"

template <int dim>
class PeriodicConfiguration : public BoundaryValues<dim>
{
public:
    PeriodicConfiguration(double k_ = 1,
                          double eps_ = 0.1,
                          double S_ = 0.6751);

    PeriodicConfiguration(std::map<std::string, boost::any> &am);

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const dealii::Point<dim> &p,
					          dealii::Vector<double> &value) const override;

    virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                            std::vector<double> &value_list,
                            const unsigned int component = 0) const override;

    virtual void
    vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                      std::vector<dealii::Vector<double>>   &value_list)
                      const override;

private:

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<BoundaryValues<dim>>(*this);
        ar & k;
        ar & eps;
        ar & S;
    }

    double k;
    double eps;
    double S;
};

BOOST_CLASS_EXPORT_KEY(PeriodicConfiguration<2>)
BOOST_CLASS_EXPORT_KEY(PeriodicConfiguration<3>)

#endif
