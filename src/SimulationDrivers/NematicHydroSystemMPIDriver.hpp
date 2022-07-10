#ifndef NEMATIC_HYDRO_SYSTEM_MPI_DRIVER_HPP
#define NEMATIC_HYDRO_SYSTEM_MPI_DRIVER_HPP

#include <deal.II/distributed/tria.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/base/parameter_handler.h>

#include <string>
#include <memory>

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "LiquidCrystalSystems/HydroSystemMPI.hpp"

template <int dim>
class NematicHydroSystemMPIDriver
{
public:
    NematicHydroSystemMPIDriver(unsigned int degree_ = 1,
                                unsigned int num_refines_ = 6,
                                double left_ = 1.0,
                                double right_ = -1.0,
                                double dt_ = 1.0,
                                unsigned int n_steps_ = 1,
                                double simulation_tol_ = 1e-10,
                                unsigned int simulation_max_iters_ = 20,
                                double update_coeff = 1.0,
                                double update_hydro_coeff = 1.0,
                                std::string data_folder_ = std::string("./"),
                                std::string nematic_config_filename_ = std::string(""),
                                std::string hydro_config_filename_ = std::string(""),
                                std::string archive_filename_
                                = std::string("nematic_hydro_simulation.ar"));

    static void declare_parameters(dealii::ParameterHandler &prm);

    void run(dealii::ParameterHandler &prm);
    void run_deserialization(dealii::ParameterHandler &prm);

private:
    void make_grid();
    void iterate_timestep(NematicSystemMPI<dim> &nematic_system);
    void iterate_timestep(NematicSystemMPI<dim> &nematic_system,
                          HydroSystemMPI<dim> &hydro_system);

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

    bool currently_coupled;

    unsigned int degree;
    unsigned int num_refines;
    double left;
    double right;

    double dt;
    unsigned int n_steps;
    double relaxation_time;

    double simulation_tol;
    unsigned int simulation_max_iters;
    double update_coeff;
    double update_hydro_coeff;

    std::string data_folder;
    std::string nematic_config_filename;
    std::string hydro_config_filename;
    std::string archive_filename;
};

#endif
