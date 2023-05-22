#ifndef NEMATIC_SYSTEM_MPI_DRIVER_HPP
#define NEMATIC_SYSTEM_MPI_DRIVER_HPP

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/hdf5.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_tools_cache.h>
#include <string>
#include <memory>

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"

template <int dim>
class NematicSystemMPIDriver
{
public:
    NematicSystemMPIDriver(unsigned int degree_ = 1,
                           unsigned int num_refines_ = 6,
                           double left_ = 1.0,
                           double right_ = -1.0,
                           std::string grid_type = "hypercube",
                           unsigned int num_further_refines_ = 0,
                           double dt_ = 1.0,
                           unsigned int n_steps_ = 1,
                           unsigned int n_recentered_steps = 0,
                           std::string time_discretization_ = std::string("convex_splitting"),
                           double simulation_tol_ = 1e-10,
                           double simulation_newton_step_ = 1.0,
                           unsigned int simulation_max_iters_ = 20,
                           double defect_size_ = 2.0,
                           double defect_charge_threshold_ = 0.3,
                           unsigned int vtu_interval_ = 10,
                           unsigned int checkpoint_interval_ = 10,
                           std::string data_folder_ = std::string("./"),
                           std::string config_filename_ = std::string(""),
                           std::string defect_filename_ = std::string(""),
                           std::string energy_filename_ = std::string(""),
                           std::string archive_filename_
                           = std::string("lc_simulation.ar"));

    void run(std::string parameter_filename);
    std::unique_ptr<NematicSystemMPI<dim>> 
        deserialize(const std::string &filename);
    dealii::GridTools::Cache<dim> get_grid_cache();
    std::vector<dealii::BoundingBox<dim>> 
        get_bounding_boxes(unsigned int refinement_level=0,
                           bool allow_merge=false,
                           unsigned int max_boxes=dealii::numbers::
                           invalid_unsigned_int);
    std::pair<std::vector<double>, std::vector<hsize_t>>
    read_configuration_at_points(const NematicSystemMPI<dim> &nematic_system,
                                 const std::vector<dealii::Point<dim>> &p,
                                 const dealii::GridTools::Cache<dim> &cache,
                                 const std::vector<std::vector<dealii::BoundingBox<dim>>>
                                 &global_bounding_boxes,
                                 hsize_t offset=0);

    static void declare_parameters(dealii::ParameterHandler &prm);

private:
    void make_grid();
    void refine_further();
    void refine_around_defects();

    void conditional_output(unsigned int timestep);

    void iterate_timestep();

    void get_parameters(dealii::ParameterHandler &prm);
    void print_parameters(std::string filename,
                          dealii::ParameterHandler &prm);

    MPI_Comm mpi_communicator;
    dealii::parallel::distributed::Triangulation<dim> tria;
    dealii::Triangulation<dim> coarse_tria;
    std::unique_ptr<NematicSystemMPI<dim>> nematic_system;

    std::vector<std::vector<double>> previous_defect_points;
    std::vector<std::vector<double>> defect_points;
    std::vector<double> defect_refine_distances;

    dealii::ConditionalOStream pcout;
    dealii::TimerOutput computing_timer;

    unsigned int degree;
    unsigned int num_refines;
    double left;
    double right;
    std::string grid_type;
    unsigned int num_further_refines;
    bool recenter_refinement;
    double defect_position;
    double defect_radius;
    double outer_radius;

    double dt;
    unsigned int n_steps;
    unsigned int n_recentered_steps;
    double theta;

    std::string time_discretization;
    double simulation_tol;
    double simulation_newton_step;
    unsigned int simulation_max_iters;
    bool freeze_defects;

    double defect_size;
    double defect_charge_threshold;

    unsigned int vtu_interval;
    unsigned int checkpoint_interval;
    std::string data_folder;
    std::string config_filename;
    std::string defect_filename;
    std::string energy_filename;
    std::string archive_filename;
};

#endif
