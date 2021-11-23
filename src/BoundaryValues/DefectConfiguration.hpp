#ifndef DEFECT_CONFIGURATION_HPP
#define DEFECT_CONFIGURATION_HPP

#include "BoundaryValues.hpp"
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

template <int dim>
class DefectConfiguration : public BoundaryValues<dim>
{
public:
    DefectConfiguration(double S_, double k_);

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component) const override;

    
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
    double S = 0.6;
    double psi = 0;
    double k = 0.5;
    dealii::Point<dim> center;
};

#endif