#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/point.h>

#include <vector>
#include <fstream>
#include <string>

int main()
{
    const int dim = 2;
    dealii::Triangulation<dim> tria;

    dealii::Point<dim> p1(-1.0, -1.0);
    dealii::Point<dim> p2(1.0, 1.0);
    std::vector<unsigned int> reps = {2, 1};

    dealii::GridGenerator::subdivided_hyper_rectangle(tria, reps, p1, p2);

    std::string filename("subdivided_grid.svg");
    std::ofstream output_file(filename);
    dealii::GridOut().write_svg(tria, output_file);

    return 0;
}
