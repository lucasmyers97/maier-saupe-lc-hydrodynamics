#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <vector>
#include <string>

/**
 * This class should be called from within some kind of a simulation object
 * (presumably a LiquidCrystalSystems object) in order to take the simulation
 * and print it out at some number of points.
 *
 * It should also facilitate reading out those points from a file which either
 * contains a list of coordinates, or just a meshgrid (the latter will be
 * implemented first, while the former can happen later).
 *
 * It should store the points at which the grid is being evaluated, but it can
 * just take a reference to dof_handler when it's actually reading out the
 * values of the fe object.
 *
 * Should rename it to "EvaluateFEObject" or something -- I'll do that later
 *
 * TODO Write constructor which can take in variables_map arguments
 * TODO Actually try to compile and test each portion
 * TODO Document each function
 * TODO Rename the class to something more apt
 * TODO Put class in its own comilation unit
 */
template <int dim>
class EvaluateFEObject
{

public:
    EvaluateFEObject(std::vector<std::string> meshgrid_names_)
        : meshgrid_names(meshgrid_names_)
    {};

    void read_grid(std::string input_filename, double dist_scale);
    void read_points(std::string input_filename, double dist_scale);
    void read_fe_at_points(const dealii::DoFHandler<dim> &dof_handler,
                           const dealii::Vector<double> &solution);

    void write_values_to_grid(std::string output_filename);
    void write_values_to_points(std::string output_filename);

  private:
    using mat = std::vector<std::vector<double>>;

    /* \brief dataset names in hdf5 files */
    std::vector<std::string> meshgrid_names;
    std::string pointlist_name;

    std::vector<dealii::Point<dim>> points;
    std::vector<dealii::Vector<double>> values;

    std::vector<mat> meshgrids;
    std::vector<int> grid_size;
};
