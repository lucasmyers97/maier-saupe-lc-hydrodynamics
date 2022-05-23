#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/point.h>

#include <iostream>
#include <fstream>

int main()
{
    const int dim = 2;
    int repetitions = 20;
    int n_fine_iters = 6;
    double left = -700.0 / 3.0;
    double right = 700.0 / 3.0;

    double coarse_length = 70.0 / 3.0;

    dealii::Triangulation<dim> tria;
    dealii::GridGenerator::subdivided_hyper_cube(tria, repetitions,
                                                 left, right);

    double fine_left = 0;
    double fine_right = 0;
    dealii::Point<dim> center;
    for (unsigned int i = 2; i < n_fine_iters + 2; ++i)
    {
        fine_left = -(i * coarse_length);
        fine_right = i * coarse_length;

        for (auto &cell : tria.active_cell_iterators())
        {
            center = cell->center();
            if ((center[0] >= fine_left) && (center[0] <= fine_right)
                && (center[1] >= fine_left) && (center[1] <= fine_right))
                cell->set_refine_flag();
        }

        tria.execute_coarsening_and_refinement();
    }

    std::ofstream out("zumer_grid.svg");
    dealii::GridOut grid_out;
    grid_out.write_svg(tria, out);

    std::cout << tria.n_active_cells() << std::endl;

    return 0;
}
