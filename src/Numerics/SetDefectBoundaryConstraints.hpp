#ifndef SET_DEFECT_BOUNDARY_CONSTRAINTS
#define SET_DEFECT_BOUNDARY_CONSTRAINTS

#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>

#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/base/quadrature.h>

#include <vector>
#include <limits>

#include "Utilities/maier_saupe_constants.hpp"

namespace SetDefectBoundaryConstraints
{

using mat_id = dealii::types::material_id;
namespace msc = maier_saupe_constants;

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
                    const std::vector<mat_id> &defect_ids,
                    double defect_radius)
{
    Assert( defect_points.size() == defect_ids.size(), 
            ExcMessage("Defect points and ID's are different sizes") )

    double cell_distance = std::numeric_limits<double>::max();

    for (const auto &cell : tria.active_cell_iterators())
    {
        if (cell->is_artificial())
            continue;

        for (std::size_t i = 0; i < defect_points.size(); ++i)
        {
            cell_distance = defect_points[i].distance( cell->center() );

            if (cell_distance < defect_radius)
                cell->set_material_id(defect_ids[i]);
        }
    }
}



/**
 * \brief Runs through triangulation, and for faces which sit at the boundary
 * between two domains, assigns values to degrees of freedom which interpolate
 * Function values given by a function map.
 *
 * Functions of two different domains must match at the interface.
 */
template <int dim>
inline void
interpolate_boundary_values
(const dealii::DoFHandler<dim> &dof,
 const std::map<mat_id, const dealii::Function<dim>* > &function_map,
 std::map<dealii::types::global_dof_index, double> &boundary_values)
{
    if (function_map.size() == 0)
        return;

    // vector to store indices
    std::vector<dealii::types::global_dof_index> face_dofs;
    face_dofs.reserve(dof.get_fe_collection().max_dofs_per_face());

    // vector to store function values
    std::vector<dealii::Vector<double>> dof_values_system;
    dof_values_system.reserve(dof.get_fe_collection().max_dofs_per_face());

    // vector to store dof locations
    std::vector<dealii::Point<dim>> dof_locations;
    dof_locations.reserve(dof.get_fe_collection().max_dofs_per_cell());

    for (const auto &cell : dof.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        const mat_id material_component = cell->material_id();
        const dealii::FiniteElement<dim> &fe = cell->get_fe();

        // assign face dof constraints if between two domains
        for (const unsigned int face_no : cell->face_indices())
        {
            if (cell->at_boundary(face_no))
                continue;

            const auto face = cell->face(face_no);
            const auto neighbor = cell->neighbor(face_no);

            if (neighbor->has_children())
                continue;

            if (material_component == neighbor->material_id())
                continue;

            if ((function_map.find(material_component) == function_map.end())
                || (fe.n_dofs_per_face() == 0))
                continue;

            const dealii::Quadrature<dim - 1> 
                q(fe.get_unit_face_support_points());
            dealii::FEFaceValues<dim> 
                fe_face_values(fe, q, dealii::update_quadrature_points);
            fe_face_values.reinit(cell, face_no);

            // get dof numbers and support points of dofs
            face_dofs.resize(fe.n_dofs_per_face(face_no));
            face->get_dof_indices(face_dofs, cell->active_fe_index());
            dof_locations = fe_face_values.get_quadrature_points();

            dof_values_system.resize(dof_locations.size());
            for (auto &dof_value : dof_values_system)
                // should reinit with function_map.find(material_component)->second->n_components
                dof_value.reinit(msc::vec_dim<dim>);
            function_map.find(material_component)
                ->second->vector_value_list(dof_locations, dof_values_system);

            for (unsigned int i = 0; i < face_dofs.size(); ++i)
            {
                unsigned int component 
                    = fe.face_system_to_component_index(i, face_no).first;
                boundary_values[face_dofs[i]] 
                    = dof_values_system[i](component);
            }
        }
    }
}

}

#endif
