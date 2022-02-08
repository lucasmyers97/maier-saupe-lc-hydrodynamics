#include "EvaluateFEObject.hpp"

#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/fe_field_function.h>

#include <vector>
#include <string>
#include <algorithm>

#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

/**
 * Reads points at which finite element object will be evaluated from hdf5 file.
 * Hdf5 file should either have a meshgrid (X and Y arrays holding x and y vals)
 * or it should be a list of points (dim x n_pts array called `points`).
 *
 * Should allow users to specify the name of the grid or points object in the
 * hdf5 file.
 *
 * Will start with 2D, but should be able to extend idea to 3D.
 *
 * Should just have a vector of names of meshgrid objects.
 */
template <int dim>
void  EvaluateFEObject<dim>::read_grid(std::string input_filename,
                                       double dist_scale)
{
    // read file
    HighFive::File f(input_filename);

    // get object info and create dataset vector
    std::vector<std::string> object_names = f.listObjectNames();

    meshgrids.resize(dim);
    for (int i = 0; i < meshgrid_names.size(); ++i)
    {
        // make sure name of meshgrid in file is specified
        if (meshgrid_names[i].empty())
        {
            throw "Tried to read in meshgrid without specifying name";
            return;
        }

        auto mesh_iter = std::find(object_names.begin(),
                                   object_names.end(),
                                   meshgrid_names[i]);

        // check whether specified name is actually in the file
        if (mesh_iter == object_names.end())
        {
          throw "No object named " + meshgrid_names[i] + " in file";
          return;
        }

        meshgrids[i] = H5Easy::load<mat>(f, *mesh_iter);
    }

    grid_size.resize(dim);
    grid_size[0] = meshgrids[0].size();
    grid_size[1] = meshgrids[0][0].size();

    int n_pts = 1;
    for (const auto& grid_dim : grid_size)
        n_pts *= grid_dim;

    points.resize(n_pts);
    for (int i = 0; i < grid_size[0]; ++i)
        for (int j = 0; j < grid_size[1]; ++j)
        {
            points[i * grid_size[1] + j](0) = dist_scale * meshgrids[0][i][j];
            points[i * grid_size[1] + j](1) = dist_scale * meshgrids[1][i][j];
        }
}



/**
 * Reads points at which finite element object will be evaluated from hdf5 file.
 * Hdf5 file should either have a meshgrid (X and Y arrays holding x and y vals)
 * or it should be a list of points (dim x n_pts array called `points`).
 *
 * Should allow users to specify the name of the grid or points object in the
 * hdf5 file.
 *
 * Will start with 2D, but should be able to extend idea to 3D.
 *
 * Should just have a vector of names of meshgrid objects.
 */
template <int dim>
void  EvaluateFEObject<dim>::read_points(std::string filename,
                                         double dist_scale)
{
    // read file
    HighFive::File f(filename);

    // get object info and create dataset vector
    std::vector<std::string> object_names = f.listObjectNames();

    if (pointlist_name.empty())
    {
        throw "Tried to read in points without specifying name";
        return;
    }

    auto points_iter = std::find(object_names.begin(),
                                 object_names.end(),
                                 pointlist_name);

    if (points_iter == object_names.end())
    {
        throw "No object named " + pointlist_name + " in file";
        return;
    }

    mat point_vec = H5Easy::load<mat>(f, *points_iter);
    points.resize(point_vec.size());

    for (int i = 0; i < points.size(); ++i)
    {
        points[i](0) = dist_scale * point_vec[i][0];
        points[i](1) = dist_scale * point_vec[i][1];
    }
}



template <int dim>
void EvaluateFEObject<dim>::read_fe_at_points(
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::Vector<double> &solution)
{
    values.resize(points.size(),
                  dealii::Vector<double>(msc::vec_dim<dim>));

    dealii::Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
    fe_function.vector_value_list(points, values);
}



template <int dim>
void  EvaluateFEObject<dim>::write_values_to_grid(std::string output_filename)
{
    // put values from fe solution into Q_vec object
    std::vector<mat> Q_vec(msc::vec_dim<dim>);
    for (int vec_idx = 0; vec_idx < msc::vec_dim<dim>; ++vec_idx)
    {
        // resize to be shape of grid
        Q_vec[vec_idx].resize(grid_size[0]);
        for (int i = 0; i < grid_size[0]; ++i) {
            Q_vec[vec_idx][i].resize(grid_size[1]);
        }
        
        // populate with values
        for (int i = 0; i < grid_size[0]; ++i) 
            for (int j = 0; j < grid_size[1]; ++j) 
                Q_vec[vec_idx][i][j] = values[i*grid_size[1] + j][vec_idx];
    }

    // Write to hdf5 file
    HighFive::File f(output_filename,
                     HighFive::File::ReadWrite |
                     HighFive::File::Create |
                     HighFive::File::Truncate);
    for (int vec_idx = 0; vec_idx < msc::vec_dim<dim>; ++vec_idx)
        H5Easy::dump(f, "Q" + std::to_string(vec_idx + 1), Q_vec[vec_idx]);

    H5Easy::dump(f, "X", meshgrids[0]);
    H5Easy::dump(f, "Y", meshgrids[1]);
}



template <int dim>
void  EvaluateFEObject<dim>::write_values_to_points(std::string output_filename)
{
    // allocate Q-vector as n_pts x vec_dim array
    mat Q_vec(points.size(),
              std::vector<double> (msc::vec_dim<dim>));

    // populate output Q-vector with values taken from FEFieldFunction
    for (int pt_idx = 0; pt_idx < points.size(); ++pt_idx)
        for (int vec_idx = 0; vec_idx < msc::vec_dim<dim>; ++vec_idx)
            Q_vec[pt_idx][vec_idx] = values[pt_idx][vec_idx];

    // Write to hdf5 file
    HighFive::File f(output_filename,
                     HighFive::File::ReadWrite |
                     HighFive::File::Create |
                     HighFive::File::Truncate);
    H5Easy::dump(f, "Q_vec", Q_vec);
}


template class EvaluateFEObject<2>;
template class EvaluateFEObject<3>;
