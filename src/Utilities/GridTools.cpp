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

#include "Numerics/NumericalTools.hpp"

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
read_configuration_at_points(const std::vector<dealii::Point<dim>> &points,
                             const dealii::DoFHandler<dim> &dof_handler,
                             const VectorType &configuration,
                             const dealii::GridTools::Cache<dim> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<dim>>>
                             &global_bounding_boxes,
                             hsize_t offset)
{
    const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();

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
            for (std::size_t k = 0; k < fe.n_components(); ++k)
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



template <int dim, typename VectorType>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_radial_points(const RadialPointSet<dim> &point_set,
                                    const MPI_Comm &mpi_communicator,
                                    const dealii::DoFHandler<dim> &dof_handler,
                                    const VectorType &configuration,
                                    const dealii::GridTools::Cache<dim> &cache,
                                    const std::vector<std::vector<dealii::BoundingBox<dim>>>
                                    &global_bounding_boxes)
{
    unsigned int this_process 
        = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    std::vector<double> r = NumericalTools::linspace(point_set.r_0, 
                                                     point_set.r_f, 
                                                     point_set.n_r);
    std::vector<double> theta = NumericalTools::linspace(0, 
                                                         2 * M_PI, 
                                                         point_set.n_theta);
    std::vector<dealii::Point<dim>> p;
    if (this_process == 0)
        p.resize(point_set.n_theta);

    std::vector<double> local_values;
    std::vector<hsize_t> local_value_indices;
    std::vector<double> total_local_values;
    std::vector<hsize_t> total_local_value_indices;
    hsize_t offset = 0;
    for (std::size_t i = 0; i < point_set.n_r; ++i)
    {
        offset = i * point_set.n_theta;

        // get points for this timestep, and this r-value
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            for (std::size_t j = 0; j < point_set.n_theta; ++j)
            {
                p[j][0] = r[i] * std::cos(theta[j]) + point_set.center[0];
                p[j][1] = r[i] * std::sin(theta[j]) + point_set.center[0];
            }

        std::tie(local_values, local_value_indices)
            = read_configuration_at_points(p,
                                           dof_handler,
                                           configuration,
                                           cache,
                                           global_bounding_boxes,
                                           offset);

        // concatenate local values corresponding to const r slice
        // to the vector holding *all* locally-held points
        total_local_values.insert(total_local_values.end(),
                                  local_values.begin(),
                                  local_values.end());
        total_local_value_indices.insert(total_local_value_indices.end(),
                                         local_value_indices.begin(),
                                         local_value_indices.end());
    }

    return std::make_pair(total_local_values, total_local_value_indices);
}



template
std::vector<dealii::BoundingBox<2>> 
get_bounding_boxes<2>(const dealii::Triangulation<2>& tria,
                      unsigned int refinement_level,
                      bool allow_merge,
                      unsigned int max_boxes);

template
std::vector<dealii::BoundingBox<3>> 
get_bounding_boxes<3>(const dealii::Triangulation<3>& tria,
                      unsigned int refinement_level,
                      bool allow_merge,
                      unsigned int max_boxes);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points<2, dealii::Vector<double>>(const std::vector<dealii::Point<2>> &points,
                                                        const dealii::DoFHandler<2> &dof_handler,
                                                        const dealii::Vector<double> &configuration,
                                                        const dealii::GridTools::Cache<2> &cache,
                                                        const std::vector<std::vector<dealii::BoundingBox<2>>>
                                                        &global_bounding_boxes,
                                                        hsize_t offset);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points<3, dealii::Vector<double>>(const std::vector<dealii::Point<3>> &points,
                                                        const dealii::DoFHandler<3> &dof_handler,
                                                        const dealii::Vector<double> &configuration,
                                                        const dealii::GridTools::Cache<3> &cache,
                                                        const std::vector<std::vector<dealii::BoundingBox<3>>>
                                                        &global_bounding_boxes,
                                                        hsize_t offset);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points<2, dealii::LinearAlgebraTrilinos::MPI::Vector>
(const std::vector<dealii::Point<2>> &points,
 const dealii::DoFHandler<2> &dof_handler,
 const dealii::LinearAlgebraTrilinos::MPI::Vector &configuration,
 const dealii::GridTools::Cache<2> &cache,
 const std::vector<std::vector<dealii::BoundingBox<2>>>
 &global_bounding_boxes,
 hsize_t offset);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points<3, dealii::LinearAlgebraTrilinos::MPI::Vector>
(const std::vector<dealii::Point<3>> &points,
 const dealii::DoFHandler<3> &dof_handler,
 const dealii::LinearAlgebraTrilinos::MPI::Vector &configuration,
 const dealii::GridTools::Cache<3> &cache,
 const std::vector<std::vector<dealii::BoundingBox<3>>>
 &global_bounding_boxes,
 hsize_t offset);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_radial_points<2, dealii::Vector<double>>(const RadialPointSet<2> &point_set,
                                                               const MPI_Comm &mpi_communicator,
                                                               const dealii::DoFHandler<2> &dof_handler,
                                                               const dealii::Vector<double> &configuration,
                                                               const dealii::GridTools::Cache<2> &cache,
                                                               const std::vector<std::vector<dealii::BoundingBox<2>>>
                                                               &global_bounding_boxes);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_radial_points<3, dealii::Vector<double>>(const RadialPointSet<3> &point_set,
                                                               const MPI_Comm &mpi_communicator,
                                                               const dealii::DoFHandler<3> &dof_handler,
                                                               const dealii::Vector<double> &configuration,
                                                               const dealii::GridTools::Cache<3> &cache,
                                                               const std::vector<std::vector<dealii::BoundingBox<3>>>
                                                               &global_bounding_boxes);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_radial_points<2, dealii::LinearAlgebraTrilinos::MPI::Vector>
(const RadialPointSet<2> &point_set,
 const MPI_Comm &mpi_communicator,
 const dealii::DoFHandler<2> &dof_handler,
 const dealii::LinearAlgebraTrilinos::MPI::Vector &configuration,
 const dealii::GridTools::Cache<2> &cache,
 const std::vector<std::vector<dealii::BoundingBox<2>>>
 &global_bounding_boxes);

template
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_radial_points<3, dealii::LinearAlgebraTrilinos::MPI::Vector>
(const RadialPointSet<3> &point_set,
 const MPI_Comm &mpi_communicator,
 const dealii::DoFHandler<3> &dof_handler,
 const dealii::LinearAlgebraTrilinos::MPI::Vector &configuration,
 const dealii::GridTools::Cache<3> &cache,
 const std::vector<std::vector<dealii::BoundingBox<3>>>
 &global_bounding_boxes);

} //GridTools
