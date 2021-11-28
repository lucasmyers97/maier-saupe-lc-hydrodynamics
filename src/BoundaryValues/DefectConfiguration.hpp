#ifndef DEFECT_CONFIGURATION_HPP
#define DEFECT_CONFIGURATION_HPP

#include "BoundaryValues.hpp"
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

enum class DefectCharge
  {
   plus_half,
   minus_half,
   plus_one,
   minus_one
  };



struct DefectConfigurationParams : public BoundaryValuesParams
{
  std::string name = "defect";
  double S = 0.6751;
  DefectCharge charge = DefectCharge::plus_half;
  double psi = 0;
  double k;

  void get_charge_from_name(const std::string charge_name)
  {
    if (charge_name == "plus_half")
      {
        charge = DefectCharge::plus_half;
      }
    else if (charge_name == "minus_half")
      {
        charge = DefectCharge::minus_half;
      }
    else if (charge_name == "plus_one")
      {
        charge = DefectCharge::plus_one;
      }
    else if (charge_name == "minus_one")
      {
        charge = DefectCharge::minus_one;
      }
  }
};



template <int dim>
class DefectConfiguration : public BoundaryValues<dim>
{
public:
    DefectConfiguration();
    DefectConfiguration(double S_, DefectCharge charge);
    DefectConfiguration(DefectConfigurationParams params);

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
    double S = 0.6751;
    DefectCharge charge = DefectCharge::plus_half;
    double psi = 0;
    double k;
    dealii::Point<dim> center;
};




#endif
