#include <boost/any.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/export.hpp>

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <fstream>

#define private public

#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"

int main()
{
    const int dim = 3;

    std::map<std::string, boost::any> am;
    am["S-value"] = 0.6751;
    am["defect-charge-name"] = std::string("plus-half-minus-half");
    am["centers"] = std::vector<double>({-5.0, 0, 5.0, 0});
    am["boundary-values-name"] = std::string("two-defect");

    std::unique_ptr<BoundaryValues<dim>> boundary_values
        = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(am);

    {
        std::ofstream ofs("boundary_pointer.txt");
        boost::archive::binary_oarchive oa(ofs);
        oa << boundary_values;
    }
    {
        std::ifstream ifs("boundary_pointer.txt");
        boost::archive::binary_iarchive ia(ifs);
        ia >> boundary_values;
    }

    std::cout << boundary_values->name << "\n";


    return 0;
}
