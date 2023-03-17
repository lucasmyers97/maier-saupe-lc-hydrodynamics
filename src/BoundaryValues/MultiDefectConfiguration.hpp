#ifndef MULTI_DEFECT_CONFIGURATION_HPP
#define MULTI_DEFECT_CONFIGURATION_HPP

#include "BoundaryValues.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/fe_field_function.h>

#include <boost/any.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/export.hpp>

#include <memory>
#include <cmath>
#include <map>
#include <vector>

#include "Utilities/maier_saupe_constants.hpp"
#include "LiquidCrystalSystems/DzyaloshinskiiSystem.hpp"

template <int dim>
class MultiDefectConfiguration : public BoundaryValues<dim>
{
public:
    MultiDefectConfiguration(const std::vector<dealii::Point<dim>> 
                             &defect_centers 
                                = std::vector<dealii::Point<dim>>(1), 
                             const std::vector<double> &defect_charges 
                                = {0.5},
                             const std::vector<double> &defect_orientations 
                                = {0.0},
                             double S0 = 0.6751,
                             double eps = 0.0,
                             unsigned int degree = 1,
                             unsigned int n_refines = 8,
                             double tol = 1e-10,
                             unsigned int max_iter = 100,
                             double newton_step = 1.0);


    MultiDefectConfiguration(std::map<std::string, boost::any> &am);

    inline double 
    value_in_defect(const dealii::Functions::FEFieldFunction<1> &dzyaloshinskii_function,
                    const dealii::Point<dim> &p,
                    double defect_orientation,
                    const unsigned int component = 0) const;

    inline void
    vector_value_in_defect(const dealii::Functions::FEFieldFunction<1> 
                           &dzyaloshinskii_function,
                           const dealii::Point<dim> &p,
                           double defect_orientation,
                           dealii::Vector<double> &value) const;

    inline double
    value_outside_defect(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const;

    inline void
    vector_value_outside_defect(const dealii::Point<dim> &p,
                                dealii::Vector<double> &value) const;

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const dealii::Point<dim> &p,
					          dealii::Vector<double> &value) const override;

    // virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
    //                         std::vector<double> &value_list,
    //                         const unsigned int component = 0) const override;

    // virtual void
    // vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
    //                   std::vector<dealii::Vector<double>>   &value_list)
    //                   const override;

    void initialize();

private:
    std::vector<dealii::Point<dim>> defect_positions;
    std::vector<double> defect_charges;
    std::vector<double> defect_orientations;
    double defect_radius;
    double S0;

    /** \brief DzyaloshinskiiSystem parameters */
    double eps;
    unsigned int degree;
    unsigned int n_refines;
    double tol;
    unsigned int max_iter;
    double newton_step;

    std::vector<std::unique_ptr<DzyaloshinskiiSystem>> dzyaloshinskii_systems;
    std::vector<std::unique_ptr<dealii::Functions::FEFieldFunction<1>>>
        dzyaloshinskii_functions;

    friend class boost::serialization::access;

    template <class Archive>
    void save(Archive &ar, const unsigned int version) const
    {
        ar & boost::serialization::base_object<BoundaryValues<dim>>(*this);
        ar & defect_positions;
        ar & defect_charges;
        ar & defect_orientations;
        ar & S0;

        ar & eps;
        ar & degree;
        ar & n_refines;
        ar & tol;
        ar & max_iter;
        ar & newton_step;
    }
    template <class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<BoundaryValues<dim>>(*this);
        ar & defect_positions;
        ar & defect_charges;
        ar & defect_orientations;
        ar & S0;

        ar & eps;
        ar & degree;
        ar & n_refines;
        ar & tol;
        ar & max_iter;
        ar & newton_step;

        initialize();
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

BOOST_CLASS_EXPORT_KEY(MultiDefectConfiguration<2>)
BOOST_CLASS_EXPORT_KEY(MultiDefectConfiguration<3>)

#endif
