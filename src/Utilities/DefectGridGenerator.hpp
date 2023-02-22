#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/base/tensor.h>

#include <vector>
#include <iostream>



namespace DefectGridGenerator
{

/** 
 *  Takes in vector of defect positions, 
 */
template<int dim>
inline void 
defect_mesh_complement(dealii::Triangulation<dim> &tria,
                       double defect_position,
                       double defect_radius,
                       double outer_radius,
                       double domain_width)
{
    dealii::Triangulation<dim> tria_1;
    dealii::Triangulation<dim> tria_2;
    double pad_top_bottom = (domain_width - outer_radius) / 2.0;
    double pad_right = (domain_width / 2.0) - defect_position - outer_radius;
    double pad_left = defect_position - outer_radius;

    dealii::Point<dim> defect_point({defect_position, 0.0});

    dealii::GridGenerator::plate_with_a_hole(tria_1,
                                             defect_radius,
                                             outer_radius,
                                             pad_top_bottom,
                                             pad_top_bottom,
                                             pad_left,
                                             pad_right,
                                             defect_point);
    
    dealii::GridGenerator::plate_with_a_hole(tria_2,
                                             defect_radius,
                                             outer_radius,
                                             pad_top_bottom,
                                             pad_top_bottom,
                                             pad_right,
                                             pad_left,
                                             -defect_point);

    dealii::GridGenerator::merge_triangulations(tria_1, tria_2, tria,
                                                /*duplicated_vertex_tolerance=*/
                                                1.0e-12,
                                                /*copy_manifold_idx=*/ true);

    for (auto &cell : tria.active_cell_iterators())
    {
        for (auto &face : cell->face_iterators())
        {
            if (face->at_boundary())
            {
                dealii::Point<dim> center = face->center();
                auto dist1 = defect_point - center;
                auto dist2 = (-defect_point) - center;

                if (std::abs(dist1[0]) < outer_radius
                    && std::abs(dist1[1]) < outer_radius)
                    face->set_manifold_id(2);
                else if (std::abs(dist2[0]) < outer_radius
                         && std::abs(dist2[1]) < outer_radius)
                    face->set_manifold_id(3);
                else
                    face->set_manifold_id(1);
            }
        }
    }
    tria.set_manifold(2, dealii::PolarManifold<dim>(defect_point));
    tria.set_manifold(3, dealii::PolarManifold<dim>(-defect_point));
}

} // DefectGridGenerator
