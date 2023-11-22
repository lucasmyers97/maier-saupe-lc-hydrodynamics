#ifndef TWISTED_TWO_DEFECT_HPP
#define TWISTED_TWO_DEFECT_HPP

#include "BoundaryValues.hpp"
#include "TwoDefectConfiguration.hpp"

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <boost/any.hpp>

#include <array>
#include <string>
#include <map>



template <int dim>
class TwistedTwoDefect : public BoundaryValues<dim>
{
public:
    enum class DefectAxis
    {
        x,
        y,
        z
    };

    TwistedTwoDefect();
    TwistedTwoDefect(std::map<std::string, boost::any> &am);

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
        ar & S0;
        ar & charge;
        ar & k;
        ar & centers;
    }

    double S0 = 0.6751;
    TwoDefectCharge charge = TwoDefectCharge::plus_half_minus_half;
    std::vector<double> k = {0.5, -0.5};
    std::vector<dealii::Point<dim>> centers = {{-5, 0}, {5, 0}};
    DefectAxis axis = DefectAxis::z;
};

BOOST_CLASS_EXPORT_KEY(TwistedTwoDefect<2>)
BOOST_CLASS_EXPORT_KEY(TwistedTwoDefect<3>)

#endif
