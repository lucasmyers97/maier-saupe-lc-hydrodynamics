#ifndef PERTURBATIVE_TWO_DEFECT_HPP
#define PERTURBATIVE_TWO_DEFECT_HPP

#include "BoundaryValues.hpp"

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <boost/program_options.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <boost/any.hpp>

#include <map>
#include <string>

enum class PerturbativeTwoDefectCharge
{
    plus_half,
    minus_half
};

enum class PerturbativeTwoDefectPosition
{
    left,
    right
};

enum class PerturbativeTwoDefectIsomorph
{
    a,
    b
};



template <int dim>
class PerturbativeTwoDefect : public BoundaryValues<dim>
{
public:
    PerturbativeTwoDefect();
    PerturbativeTwoDefect(double S_, PerturbativeTwoDefectCharge charge);
    PerturbativeTwoDefect(std::map<std::string, boost::any> &am);
    PerturbativeTwoDefect(boost::program_options::variables_map vm);

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
    double calc_phi(const double r, const double theta) const;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<BoundaryValues<dim>>(*this);
        ar & S0;
        ar & eps;
        ar & d;
        ar & charge;
        ar & k;
        ar & center;
    }

    double S0 = 0.6751;
    double eps = 0.0;
    double d = 10.0;
    PerturbativeTwoDefectCharge charge = PerturbativeTwoDefectCharge::plus_half;
    PerturbativeTwoDefectPosition pos = PerturbativeTwoDefectPosition::left;
    PerturbativeTwoDefectIsomorph isomorph = PerturbativeTwoDefectIsomorph::a;
    dealii::Point<dim> center;

    double k;
    double isolated_sign = 1.0;
    double isomorph_sign = 1.0;
};

BOOST_CLASS_EXPORT_KEY(PerturbativeTwoDefect<2>)
BOOST_CLASS_EXPORT_KEY(PerturbativeTwoDefect<3>)

#endif
