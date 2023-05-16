#ifndef GRID_TOOLS_HPP
#define GRID_TOOLS_HPP

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/hdf5.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/dofs/dof_handler.h>

#include <vector>

namespace GridTools
{
/**
 * Creates a collection of bounding boxes for which all locally-owned cells
 * are completely contained in the collection.
 * Note this is a *local operation* so that each process will have its own
 * collection, and that collection will only cover the locally-owned cells on
 * that particular process.
 */
template <int dim>
std::vector<dealii::BoundingBox<dim>> 
get_bounding_boxes(const dealii::Triangulation<dim>& tria,
                   unsigned int refinement_level,
                   bool allow_merge,
                   unsigned int max_boxes);



/**
 * Calculates the value of a finite element field at some number of points.
 * This function works for a distributed system, and so the `points` input
 * vector may be different for each process.
 *
 * The return value contains information about the field at all of the points
 * from the entire pool (that is, the union of the `points` vectors on all 
 * processes) which are locally owned.
 *
 * The first entry of the return value is a vector of all of the components
 * of the finite element field at each point.
 * If f_ij is the value of the finite element field evaluated for component
 * i and point j, and there are n components and p points, the vector takes the
 * order [f_11, f_21, ..., f_n1, f12, ..., f_np]
 *
 * The second entry contains a flattened list of zero-based indices for the 
 * previous vector
 * That is, if there are n-components, it looks like: 
 * [0, 0, 0, 1, ..., 0, (n - 1), 1, 0, 1, 1, ..., (p - 1), (n - 1)]
 * However, we also include the option to pass in an offset in case
 * the passed-in points are some subset of a larger number of points so that
 * if x is the offset, the index set would look like:
 * [x, 0, x, 1, ..., x, (n - 1), (1 + x), 0, (1 + x), 1, ..., (p - 1) + x, (n - 1)]
 */
template <int dim, typename VectorType>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_points(const std::vector<dealii::Point<dim>> &points,
                             const dealii::DoFHandler<dim> &dof_handler,
                             const VectorType &configuration,
                             const dealii::GridTools::Cache<dim> &cache,
                             const std::vector<std::vector<dealii::BoundingBox<dim>>>
                             &global_bounding_boxes,
                             hsize_t offset = 0);



/**
 * Describes a set of points which are uniformly distributed around a point
 * in the radial and azimuthal directions.
 */
template <int dim>
struct RadialPointSet
{
    const dealii::Point<dim> &center;
    double r_0;
    double r_f;
    unsigned int n_r;
    unsigned int n_theta;
};



/**
 * Takes in radial point set which describes a set of points evenly distributed
 * in the azimuthal and radial direction around a point and queries the finite
 * element field at those points.
 * Returns values of each of the components of the field at those points, as
 * well as indices for use in writing to an hdf5 file.
 * See `read_configuration_at_points` for exact description of the return 
 * values.
 */
template <int dim, typename VectorType>
std::pair<std::vector<double>, std::vector<hsize_t>>
read_configuration_at_radial_points(const RadialPointSet<dim> &point_set,
                                    const dealii::DoFHandler<dim> &dof_handler,
                                    const VectorType &configuration,
                                    const dealii::GridTools::Cache<dim> &cache,
                                    const std::vector<std::vector<dealii::BoundingBox<dim>>>
                                    &global_bounding_boxes);
} //GridTools

#endif
