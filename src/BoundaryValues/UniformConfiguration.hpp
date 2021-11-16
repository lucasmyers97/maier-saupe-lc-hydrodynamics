#ifndef UNIFORM_CONFIGURATION_HPP
#define UNIFORM_CONFIGURATION_HPP

#include "BoundaryValues.hpp"
#include <deal.II/base/point.h>
#include <string>

template <int dim>
class UniformConfiguration : public BoundaryValues<dim>
{
public:
    UniformConfiguration(double S_, double phi_);

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
    double S = 0;
    double phi = 0;
};

#endif