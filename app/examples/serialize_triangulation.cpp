#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <fstream>
#include <iostream>
#include <memory>

#include <deal.II/grid/grid_generator.h>

#define protected public
#include <deal.II/grid/tria.h>

int main() {

    constexpr int dim = 2;
    dealii::Triangulation<dim> tria;

    dealii::GridGenerator::hyper_cube(tria);
    tria.refine_global(4);

    std::string filename = "serialize_triangulation.dat";
    std::ofstream ofs(filename);
    {
        boost::archive::binary_oarchive oa(ofs);
        oa << tria;
    }

    dealii::Triangulation<dim> new_tria;
    {
        std::ifstream ifs(filename);
        boost::archive::binary_iarchive ia(ifs);
        ia >> new_tria;
    }

    return 0;
}
