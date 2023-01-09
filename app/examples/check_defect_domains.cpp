#include "Numerics/SetDefectBoundaryConstraints.hpp"

#include <deal.II/base/types.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <vector>

int main()
{
    const int dim = 2;
    double left = -5.0;
    double right = 5.0;
    unsigned int n_refines = 6;

    unsigned int n_defect_points = 3;
    double defect_radius = 1.0;

    // generate triangulation
    dealii::Triangulation<dim> tria;
    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(n_refines);

    std::vector<dealii::Point<dim>> defect_points(n_defect_points);
    defect_points[0] = dealii::Point<dim>({-2.5, 0});
    defect_points[1] = dealii::Point<dim>({2.5, 0});
    defect_points[2] = dealii::Point<dim>({0, 2.5});

    std::vector<dealii::types::material_id> defect_ids(n_defect_points);
    defect_ids[0] = 1;
    defect_ids[1] = 2;
    defect_ids[2] = 3;

    SetDefectBoundaryConstraints::mark_defect_domains(tria, 
                                                      defect_points, 
                                                      defect_ids, 
                                                      defect_radius);

    // write grid to svg
    dealii::GridOut grid_out;
    dealii::GridOutFlags::Svg flags;
    flags.coloring = dealii::GridOutFlags::Svg::Coloring::material_id;
    grid_out.set_flags(flags);

    std::ofstream os("grid_out.svg");
    grid_out.write_svg(tria, os);

    return 0;
}
