/**
 * This program demonstrates the use of a std::map of boost::any objects to
 * construct a BoundaryValues object from the BoundaryValuesFactory method.
 * Note that this works equally well with any of the BoundaryValues objects,
 * provided the map has the proper arguments.
 */

#include "BoundaryValues/BoundaryValuesFactory.hpp"

#include <boost/any.hpp>

#include <memory>
#include <string>
#include <map>

int main()
{
    const int dim = 2;
    std::string boundary_values_name = "uniform";
    double S = 0.6751;
    double phi = 0;

    std::map<std::string, boost::any> am;
    am["S-value"] = S;
    am["phi"] = phi;

    std::unique_ptr<BoundaryValues<dim>> boundary_values
        = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(boundary_values_name, am);

    return 0;
}
