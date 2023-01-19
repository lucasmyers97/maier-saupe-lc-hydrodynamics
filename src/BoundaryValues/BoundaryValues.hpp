#ifndef BOUNDARY_VALUES_HPP
#define BOUNDARY_VALUES_HPP

#include "BoundaryValuesInterface.hpp"

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <deal.II/base/function.h>

#include <string>
#include "Utilities/maier_saupe_constants.hpp"


template<int dim>
class BoundaryValues : public dealii::Function<dim>
{
public:
    virtual ~BoundaryValues() = default;
    std::string name;
    std::string boundary_condition; // Dirichlet or Neumann
    std::vector<dealii::Point<dim>> defect_pts;

    BoundaryValues(std::string name_ = std::string("uniform"),
                   std::string boundary_condition_ = std::string("Dirichlet"),
                   std::vector<dealii::Point<dim>> defect_pts_ 
                   = std::vector<dealii::Point<dim>>())
        : dealii::Function<dim>(maier_saupe_constants::vec_dim<dim>)
        , name(name_)
        , boundary_condition(boundary_condition_)
        , defect_pts(defect_pts_)
    {}

  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<dealii::Function<dim>>(*this);
        ar & name;
        ar & boundary_condition;
        // ar & defect_pts;
    }
};

BOOST_CLASS_EXPORT_KEY(BoundaryValues<2>)
BOOST_CLASS_EXPORT_KEY(BoundaryValues<3>)

#endif
