#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataSpace.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Utilities/maier_saupe_constants.hpp"
#include "Postprocessors/EvaluateFEObject.hpp"

namespace msc = maier_saupe_constants;

int main()
{
    const int dim = 2;
    std::vector<std::string> meshgrid_names(dim);
    meshgrid_names[0] = "X";
    meshgrid_names[1] = "Y";

    EvaluateFEObject<dim> defect_to_grid(meshgrid_names);
    defect_to_grid.run();

    return 0;
}
