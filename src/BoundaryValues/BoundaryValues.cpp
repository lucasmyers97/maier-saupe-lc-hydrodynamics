#include "BoundaryValues.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

template class BoundaryValues<2>;
template class BoundaryValues<3>;

BOOST_CLASS_EXPORT_IMPLEMENT(BoundaryValues<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(BoundaryValues<3>)
