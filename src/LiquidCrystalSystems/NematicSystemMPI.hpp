#ifndef NEMATIC_SYSTEM_MPI_HPP
#define NEMATIC_SYSTEM_MPI_HPP

#include <deal.II/base/mpi.h>

#include <deal.II/lac/generic_linear_algebra.h>
namespace LA = dealii::LinearAlgebraTrilinos;

#include <deal.II/base/index_set.h>

#include <boost/program_options.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/vector.h>

#include <deal.II/base/parameter_handler.h>

#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/DefectConfiguration.hpp"
#include "BoundaryValues/UniformConfiguration.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"

#include <memory>
#include <string>

// Need to forward-declare coupler so it can be a friend class
template<int dim>
class NematicHydroMPICoupler;

template <int dim>
class NematicSystemMPI
{
public:
    NematicSystemMPI(const dealii::parallel::distributed::Triangulation<dim>
                     &triangulation,
                     unsigned int degree = 1,
                     std::string boundary_values_name
                     = std::string("uniform"),
                     const std::map<std::string, boost::any> &am
                     = std::map<std::string, boost::any>(
                         { {std::string("boundary-values-name"), 
                            std::string("uniform")},
                           {std::string("S-value"), 0.6751},
                           {std::string("phi"), 0.0} }
                         ),
                     double maier_saupe_alpha_ = 8.0,
                     double L2_ = 0.0,
                     double L3_ = 0.0,
                     int order = 590,
                     double lagrange_step_size = 1.0,
                     double lagrange_tol = 1e-10,
                     unsigned int lagrange_max_iters = 20);

    static void declare_parameters(dealii::ParameterHandler &prm);
    void get_parameters(dealii::ParameterHandler &prm);

    void setup_dofs(const MPI_Comm &mpi_communicator,
                    const bool initial_step);
    void initialize_fe_field(const MPI_Comm &mpi_communicator);

    void assemble_system(const double dt);
    void assemble_system_anisotropic(double dt);
    void solve_and_update(const MPI_Comm &mpi_communicator, const double alpha);
    double return_norm();
    double return_linfty_norm();
    void set_past_solution_to_current(const MPI_Comm &mpi_communicator);
    void find_defects(double min_dist, 
                      double charge_threshold,
                      double current_time);
    void calc_energy(const MPI_Comm &mpi_communicator,
                     double current_time);
    void output_defect_positions(const MPI_Comm &mpi_communicator,
                                 const std::string data_folder,
                                 const std::string filename);
    void output_configuration_energies(const MPI_Comm &mpi_communicator,
                                       const std::string data_folder,
                                       const std::string filename);

    void output_results(const MPI_Comm &mpi_communicator,
                        const dealii::parallel::distributed::Triangulation<dim>
                        &triangulation,
                        const std::string data_folder,
                        const std::string filename,
                        const int timestep) const;
    void 
    output_Q_components(const MPI_Comm &mpi_communicator,
                        const dealii::parallel::distributed::Triangulation<dim>
                        &triangulation,
                        const std::string data_folder,
                        const std::string filename,
                        const int timestep) const;

    void output_rhs_components
        (const MPI_Comm &mpi_communicator,
         const dealii::parallel::distributed::Triangulation<dim>
         &triangulation,
         const std::string data_folder,
         const std::string filename,
         const int timestep) const;

    const dealii::DoFHandler<dim>& return_dof_handler() const;
    const LA::MPI::Vector& return_current_solution() const;
    const dealii::AffineConstraints<double>& return_constraints() const;
    double return_parameters() const;
    void set_current_solution(const MPI_Comm &mpi_communicator,
                              const LA::MPI::Vector &distributed_solution);

  private:
    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;

    /** \brief Connects mesh and FE functions to problem vector and matrix */
    dealii::DoFHandler<dim> dof_handler;
    /** \brief Takes care of values associated with FE basis functions */
    dealii::FESystem<dim> fe;
    /** \brief Matrix for the linear FE problem */
    LA::MPI::SparseMatrix system_matrix;

    /** \brief Takes care of assigning boundary values to FE vector */
    dealii::AffineConstraints<double> constraints;
    /** \brief Function which is evaluated at boundary to give Dirichlet vals */
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    /** \brief FE vector holding solution from previous timestep */
    LA::MPI::Vector past_solution;
    /** \brief FE vector holding current solution iteration */
    LA::MPI::Vector current_solution;
    /** \brief Update vector for Newton-Rhapson method */
    LA::MPI::Vector system_rhs;

    /** \brief Object which handles Lagrange Multiplier inversion of Q-tensor */
    LagrangeMultiplierAnalytic<dim> lagrange_multiplier;

    /** \brief Alpha constant for bulk energy for the Maier-Saupe field theory*/
    double maier_saupe_alpha;
    double L2;
    double L3;

    /** \brief vector holding t and spatial coorinates of defect points */
    std::vector<std::vector<double>> defect_pts;

    /** \brief vector holding time values, as well as each of the energy
     *  terms including the mean field term, the entropy term, and the elastic
     *  terms in ascending order */
    std::vector<std::vector<double>> energy_vals;

    // Allows coupling between hydro and nematic systems
    friend class NematicHydroMPICoupler<dim>;

    // This gives boost::serialization access to private members
    friend class boost::serialization::access;

    /**
     * \brief Serializes class so that it can be read back into a different file
     */
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        // ar & boundary_value_func;
        ar & lagrange_multiplier;
        ar & maier_saupe_alpha;
    }
};

#endif
