#ifndef INITIAL_VALUES_HPP 
#define INITIAL_VALUES_HPP

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
class InitialValues : public dealii::Function<dim>
{
public:
    virtual ~InitialValues() = default;
    std::string name;

    InitialValues(std::string name_ = std::string("uniform"))
        : dealii::Function<dim>(maier_saupe_constants::vec_dim<dim>)
        , name(name_)
    {}

  private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::base_object<dealii::Function<dim>>(*this);
        ar & name;
    }
};

BOOST_CLASS_EXPORT_KEY(InitialValues<2>)
BOOST_CLASS_EXPORT_KEY(InitialValues<3>)

#endif
