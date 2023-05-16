#include "GridTools.hpp"

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/hdf5.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/fe/fe.h>

#include <deal.II/dofs/dof_handler.h>

#include <vector>

namespace GridTools
{

template <int dim>
std::vector<dealii::BoundingBox<dim>> 
get_bounding_boxes(const dealii::Triangulation<dim>& tria,
                   unsigned int refinement_level,
                   bool allow_merge,
                   unsigned int max_boxes)
{
    std::function<bool(const typename dealii::Triangulation<dim>::
                       active_cell_iterator &)>
        predicate_function = [](const typename dealii::Triangulation<dim>::
                                active_cell_iterator &cell)
        { return cell->is_locally_owned(); };

    return dealii::GridTools::
           compute_mesh_predicate_bounding_box(tria, 
                                               predicate_function,
                                               refinement_level,
                                               allow_merge,
                                               max_boxes);
}



template <int dim, typename VectorType>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points(const dealii::DoFHandler<dim> &dof_handler,
                             const VectorType configuration,
                             const std::vector<dealii::Point<dim>> &points,
                             const dealii::GridTools::Cache<dim> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<dim>>>
                             &global_bounding_boxes,
                             hsize_t offset)
{
    dealii::FiniteElement<dim> &fe = dof_handler.get_fe();

    using tria_cell_type = typename dealii::Triangulation<dim>::active_cell_iterator;
    using dof_cell_type = typename dealii::DoFHandler<dim>::active_cell_iterator;

    std::vector<tria_cell_type> cells;
    std::vector<std::vector<dealii::Point<dim>>> qpoints;
    std::vector<std::vector<unsigned int>> maps;
    std::vector<std::vector<dealii::Point<dim>>> local_points;
    std::vector<std::vector<unsigned int>> owners;

    std::tie(cells, qpoints, maps, local_points, owners)
        = dealii::GridTools::
          distributed_compute_point_locations(cache,
                                              points, 
                                              global_bounding_boxes);

    // go through local cells and get values there
    std::vector<double> local_values;
    std::vector<hsize_t> local_value_indices;
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        if (!cells[i]->is_locally_owned())
            continue;

        std::size_t n_q = qpoints[i].size();

        dealii::Quadrature<dim> quad(qpoints[i]);
        dealii::FEValues<dim> fe_values(fe, quad, dealii::update_values);

        
        dof_cell_type dof_cell(&dof_handler.get_triangulation(),
                               cells[i]->level(),
                               cells[i]->index(),
                               &dof_handler);
        fe_values.reinit(dof_cell);

        std::vector<dealii::Point<dim>> cell_points(n_q);
        std::vector<dealii::Vector<double>>
            cell_values(n_q, dealii::Vector<double>(fe.n_components()));
        for (std::size_t j = 0; j < n_q; ++j)
            cell_points[j] = local_points[i][j];

        fe_values.get_function_values(configuration, cell_values);

        for (std::size_t j = 0; j < n_q; ++j)
            for (std::size_t k = 0; k < fe.n_components; ++k)
            {
                local_values.push_back(cell_values[j][k]);
                // offset is in case we are writing a subset of a bigger list
                // of points
                local_value_indices.push_back(maps[i][j] + offset);
                local_value_indices.push_back(k);
            }
    }
    return std::make_pair(local_values, local_value_indices);
}


template <>
std::vector<dealii::BoundingBox<2>> 
get_bounding_boxes<2>(const dealii::Triangulation<2>& tria,
                      unsigned int refinement_level,
                      bool allow_merge,
                      unsigned int max_boxes);

template <>
std::vector<dealii::BoundingBox<3>> 
get_bounding_boxes<3>(const dealii::Triangulation<3>& tria,
                      unsigned int refinement_level,
                      bool allow_merge,
                      unsigned int max_boxes);

template <>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points(const dealii::DoFHandler<2> &dof_handler,
                             const dealii::Vector<double> configuration,
                             const std::vector<dealii::Point<2>> &points,
                             const dealii::GridTools::Cache<2> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<2>>>
                             &global_bounding_boxes,
                             hsize_t offset);

template <>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points(const dealii::DoFHandler<3> &dof_handler,
                             const dealii::Vector<double> configuration,
                             const std::vector<dealii::Point<3>> &points,
                             const dealii::GridTools::Cache<3> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<3>>>
                             &global_bounding_boxes,
                             hsize_t offset);

template <>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points(const dealii::DoFHandler<2> &dof_handler,
                             const dealii::LinearAlgebraTrilinos::MPI::Vector configuration,
                             const std::vector<dealii::Point<2>> &points,
                             const dealii::GridTools::Cache<2> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<2>>>
                             &global_bounding_boxes,
                             hsize_t offset);

template <>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points(const dealii::DoFHandler<3> &dof_handler,
                             const dealii::LinearAlgebraTrilinos::MPI::Vector configuration,
                             const std::vector<dealii::Point<3>> &points,
                             const dealii::GridTools::Cache<3> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<3>>>
                             &global_bounding_boxes,
                             hsize_t offset);

} //GridTools
