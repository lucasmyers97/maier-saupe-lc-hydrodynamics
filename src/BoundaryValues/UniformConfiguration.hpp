#ifndef UNIFORM_CONFIGURATION_HPP
#define UNIFORM_CONFIGURATION_HPP

#include "BoundaryValues.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/any.hpp>

#include <deal.II/base/point.h>

#include <string>
#include <map>


template <int dim>
class UniformConfiguration : public BoundaryValues<dim>
{
public:
    UniformConfiguration();
    UniformConfiguration(double S_, double phi_);
    UniformConfiguration(std::map<std::string, boost::any> &am);
    UniformConfiguration(boost::program_options::variables_map vm);

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
      ar & S;
      ar & phi;
    }

    double S = 0;
    double phi = 0;
};

BOOST_CLASS_EXPORT_KEY(UniformConfiguration<2>)
BOOST_CLASS_EXPORT_KEY(UniformConfiguration<3>)

#endif
