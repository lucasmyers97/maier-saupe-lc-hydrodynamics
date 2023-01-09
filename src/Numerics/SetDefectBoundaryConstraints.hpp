#ifndef SET_DEFECT_BOUNDARY_CONSTRAINTS
#define SET_DEFECT_BOUNDARY_CONSTRAINTS

#include <deal.II/base/types.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>

#include <vector>
#include <limits>

namespace SetDefectBoundaryConstraints
{

/**
 * \brief Takes a triangulation, set of defect points, and set of defect ids,
 * as well as the defect radius, and marks all cells with baricenter within
 * `defect_radius` of each defect point with the corresponding defect id.
 * All points not within `defect_radius` of a defect point are marked 0.
 */
template <int dim>
inline void 
mark_defect_domains(dealii::Triangulation<dim> &tria,
                    const std::vector<dealii::Point<dim>> &defect_points,
                    const std::vector<dealii::types::material_id> &defect_ids,
                    double defect_radius)
{
    Assert( defect_points.size() == defect_ids.size(), 
            ExcMessage("Defect points and ID's are different sizes") )

    double cell_distance = std::numeric_limits<double>::max();

    for (const auto &cell : tria.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (std::size_t i = 0; i < defect_points.size(); ++i)
        {
            cell_distance = defect_points[i].distance( cell->center() );

            if (cell_distance < defect_radius)
                cell->set_material_id(defect_ids[i]);
        }
    }
}

}

#endif
