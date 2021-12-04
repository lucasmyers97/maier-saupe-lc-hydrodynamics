#ifndef DEFECT_CONFIGURATION_HPP
#define DEFECT_CONFIGURATION_HPP

#include "BoundaryValues.hpp"

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

enum class DefectCharge
{
 plus_half,
 minus_half,
 plus_one,
 minus_one
};



template <int dim>
class DefectConfiguration : public BoundaryValues<dim>
{
public:
    DefectConfiguration();
    DefectConfiguration(double S_, DefectCharge charge);
    DefectConfiguration(boost::program_options::variables_map vm);

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
        ar & charge;
        ar & psi;
        ar & k;
        ar & center;
    }

    double S = 0.6751;
    DefectCharge charge = DefectCharge::plus_half;
    double psi = 0;
    double k;
    dealii::Point<dim> center;
};

#endif
