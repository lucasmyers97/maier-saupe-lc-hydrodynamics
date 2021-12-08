#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <fstream>
#include <iostream>
#include <memory>

#include <deal.II/grid/grid_generator.h>

#define protected public
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>

int main() {

  constexpr int dim = 2;
  dealii::Triangulation<dim> tria;

  dealii::GridGenerator::hyper_cube(tria);
  tria.refine_global(4);

  dealii::DoFHandler<dim> dof_handler(tria);

  std::string filename = "serialize_dof_handler.dat";
  std::ofstream ofs(filename);
  {
    boost::archive::binary_oarchive oa(ofs);
    oa << tria;
    oa << dof_handler;
  }

  dealii::Triangulation<dim> new_tria;
  dealii::DoFHandler<dim> new_dof_handler(new_tria);
  {
    std::ifstream ifs(filename);
    boost::archive::binary_iarchive ia(ifs);
    ia >> new_tria;
    ia >> new_dof_handler;
  }

  return 0;
}
