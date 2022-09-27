#include "Utilities/DefectGridGenerator.hpp"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_out.h>

#include <vector>
#include <fstream>

int main()
{
    const int dim = 2;

    dealii::Triangulation<dim> tria;
    dealii::Point<dim> x1 = {20.0, 0};
    dealii::Point<dim> x2 = {-20.0, 0};
    std::vector<dealii::Point<dim>> defect_positions = {x1, x2};

    const double defect_position = 20.0;
    const double defect_radius = 2.5;
    const double outer_radius = 4 * defect_radius;
    const double domain_width = 60.0;

    DefectGridGenerator::defect_mesh_complement(tria, 
                                                defect_position,
                                                defect_radius,
                                                outer_radius,
                                                domain_width);

    dealii::GridOut grid_out;
    dealii::GridOutFlags::Svg grid_flags;
    grid_flags.label_boundary_id = true;
    grid_out.set_flags(grid_flags);
    std::ofstream ofs("grid_out.svg");
    grid_out.write_svg(tria, ofs);

    return 0;
}
