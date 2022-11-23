#ifndef DZYALOSHINSKII_FUNCTION_HPP
#define DZYALOSHINSKII_FUNCTION_HPP

#include "BoundaryValues.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/fe_field_function.h>

#include <boost/any.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <memory>
#include <cmath>
#include <map>

#include "Utilities/maier_saupe_constants.hpp"
#include "LiquidCrystalSystems/DzyaloshinskiiSystem.hpp"

template <int dim>
class DzyaloshinskiiFunction : public BoundaryValues<dim>
{
public:
    DzyaloshinskiiFunction(const dealii::Point<dim> &p = dealii::Point<dim>(), 
                           double S0_ = 0.6751,
                           double eps_ = 0.0,
                           unsigned int degree_ = 1,
                           double charge_ = 0.5,
                           unsigned int n_refines_ = 8,
                           double tol_ = 1e-10,
                           unsigned int max_iter_ = 100,
                           double newton_step_ = 1.0);


    DzyaloshinskiiFunction(std::map<std::string, boost::any> &am);

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

    void initialize();

private:
    dealii::Point<dim> defect_center;
    double S0;

    /** \brief DzyaloshinskiiSystem parameters */
    double eps;
    unsigned int degree;
    double charge;
    unsigned int n_refines;
    double tol;
    unsigned int max_iter;
    double newton_step;

    std::unique_ptr<DzyaloshinskiiSystem> dzyaloshinskii_system;
    std::unique_ptr<dealii::Functions::FEFieldFunction<1>> 
        dzyaloshinskii_function;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<BoundaryValues<dim>>(*this);
        ar & defect_center;
        ar & S0;
    }
};

BOOST_CLASS_EXPORT_KEY(DzyaloshinskiiFunction<2>)
BOOST_CLASS_EXPORT_KEY(DzyaloshinskiiFunction<3>)

#endif
