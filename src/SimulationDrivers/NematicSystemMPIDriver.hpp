#ifndef NEMATIC_SYSTEM_MPI_DRIVER_HPP
#define NEMATIC_SYSTEM_MPI_DRIVER_HPP

#include <deal.II/distributed/tria.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/base/parameter_handler.h>

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
                           double dt_ = 1.0,
                           unsigned int n_steps_ = 1,
                           double simulation_tol_ = 1e-10,
                           unsigned int simulation_max_iters_ = 20,
                           double defect_size_ = 2.0,
                           double defect_charge_threshold_ = 0.3,
                           std::string data_folder_ = std::string("./"),
                           std::string config_filename_ = std::string(""),
                           std::string defect_filename_ = std::string(""),
                           std::string archive_filename_
                           = std::string("lc_simulation.ar"));

    void run(std::string parameter_filename);
    void run_deserialization();

    static void declare_parameters(dealii::ParameterHandler &prm);

private:
    void make_grid();
    void make_fine_grid();
    void iterate_timestep(NematicSystemMPI<dim> &lc_system);

    void get_parameters(dealii::ParameterHandler &prm);
    void print_parameters(std::string filename,
                          dealii::ParameterHandler &prm);

    void serialize_nematic_system(const NematicSystemMPI<dim> &nematic_system,
                                  const std::string filename);
    std::unique_ptr<NematicSystemMPI<dim>>
    deserialize_nematic_system(const std::string filename);

    MPI_Comm mpi_communicator;
    dealii::parallel::distributed::Triangulation<dim> tria;
    dealii::Triangulation<dim> coarse_tria;

    dealii::ConditionalOStream pcout;
    dealii::TimerOutput computing_timer;

    unsigned int degree;
    unsigned int num_refines;
    double left;
    double right;

    double dt;
    unsigned int n_steps;

    double simulation_tol;
    unsigned int simulation_max_iters;

    double defect_size;
    double defect_charge_threshold;

    std::string data_folder;
    std::string config_filename;
    std::string defect_filename;
    std::string archive_filename;
};

#endif
