#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <iostream>

int main(int ac, char* av[])
{
    constexpr int dim = 2;
    dealii::Triangulation<dim> tria;
    dealii::GridGenerator::hyper_ball_balanced(tria);
    unsigned int n_refines = std::stoi(std::string(av[1]));
    tria.refine_global(n_refines);
    for (const auto &cell : tria.active_cell_iterators())
        std::cout << cell->level() << "\n";

    return 0;
}
