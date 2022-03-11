#ifndef TWO_DEFECT_CONFIGURATION_HPP
#define TWO_DEFECT_CONFIGURATION_HPP

#include "BoundaryValues.hpp"

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

#include <array>

enum class TwoDefectCharge
{
 plus_half_minus_half,
 plus_half_minus_half_alt,
};



template <int dim>
class TwoDefectConfiguration : public BoundaryValues<dim>
{
public:
    TwoDefectConfiguration();
    TwoDefectConfiguration(double S_, TwoDefectCharge charge);
    TwoDefectConfiguration(boost::program_options::variables_map vm);

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
        ar & k;
        ar & centers;
    }

    double S = 0.6751;
    TwoDefectCharge charge = TwoDefectCharge::plus_half_minus_half;
    std::vector<double> k = {0.5, -0.5};
    std::vector<dealii::Point<dim>> centers = {{-5, 0}, {5, 0}};
};

#endif
