#ifndef NEMATIC_SYSTEM_MPI_HPP
#define NEMATIC_SYSTEM_MPI_HPP

#include <deal.II/base/mpi.h>

#include <deal.II/base/types.h>
#include <deal.II/lac/generic_linear_algebra.h>
namespace LA = dealii::LinearAlgebraTrilinos;

#include <deal.II/base/index_set.h>

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>

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
#include <map>

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
                           {std::string("boundary-condition"),
                            std::string("Dirichlet")},
                           {std::string("S-value"), 0.6751},
                           {std::string("phi"), 0.0} }
                         ),
                     double maier_saupe_alpha_ = 8.0,
                     double L2_ = 0.0,
                     double L3_ = 0.0,
                     double A_ = -0.064,
                     double B_ = -1.57,
                     double C_ = 1.29,
                     std::string field_theory_ = std::string("MS"),
                     int order = 590,
                     double lagrange_step_size = 1.0,
                     double lagrange_tol = 1e-10,
                     unsigned int lagrange_max_iters = 20);

    NematicSystemMPI(unsigned int degree,
                     const std::string& field_theory,
                     double L2,
                     double L3,

                     double maier_saupe_alpha,

                     LagrangeMultiplierAnalytic<dim>&& lagrange_multiplier,

                     double A,
                     double B,
                     double C,

                     std::map<dealii::types::boundary_id, std::unique_ptr<BoundaryValues<dim>>> boundary_value_funcs,
                     std::unique_ptr<BoundaryValues<dim>> initial_value_func,
                     std::unique_ptr<BoundaryValues<dim>> left_internal_boundary_func,
                     std::unique_ptr<BoundaryValues<dim>> right_internal_boundary_func);

    static void declare_parameters(dealii::ParameterHandler &prm);
    void get_parameters(dealii::ParameterHandler &prm);

    void reinit_dof_handler(const dealii::Triangulation<dim> &tria);
    void setup_dofs(const MPI_Comm &mpi_communicator, const bool grid_modified);
    void setup_dofs(const MPI_Comm &mpi_communicator, 
                    dealii::Triangulation<dim> &tria,
                    double fixed_defect_radius);

    void initialize_fe_field(const MPI_Comm &mpi_communicator);
    void initialize_fe_field(const MPI_Comm &mpi_communicator,
                             LA::MPI::Vector &locally_owned_solution);

    void assemble_system(double dt, double theta, std::string &time_discretization);
    void solve_and_update(const MPI_Comm &mpi_communicator, const double alpha);
    double return_norm();
    double return_linfty_norm();
    void set_past_solution_to_current(const MPI_Comm &mpi_communicator);
    std::vector<std::vector<double>> find_defects(double min_dist, 
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
                        const int timestep);
    void 
    output_Q_components(const MPI_Comm &mpi_communicator,
                        const dealii::parallel::distributed::Triangulation<dim>
                        &triangulation,
                        const std::string data_folder,
                        const std::string filename,
                        const int timestep) const;

    const dealii::DoFHandler<dim>& return_dof_handler() const;
    const LA::MPI::Vector& return_current_solution() const;
    const LA::MPI::Vector& return_past_solution() const;
    const LA::MPI::Vector& return_residual() const;
    const dealii::AffineConstraints<double>& return_constraints() const;
    double return_parameters() const;
    const std::vector<dealii::Point<dim>> &return_initial_defect_pts() const;
    void set_current_solution(const MPI_Comm &mpi_communicator,
                              const LA::MPI::Vector &distributed_solution);
    void set_past_solution(const MPI_Comm &mpi_communicator,
                           const LA::MPI::Vector &distributed_solution);

    const std::vector<std::vector<double>>& get_energy_vals();
    const std::vector<std::vector<double>>& get_defect_pts();

    void set_energy_vals(const std::vector<std::vector<double>> &energy);
    void set_defect_pts(const std::vector<std::vector<double>> &defects);

    void perturb_configuration_with_director(const MPI_Comm& mpi_communicator,
                                             const dealii::DoFHandler<dim> &director_dof_handler,
                                             const LA::MPI::Vector &director_perturbation);

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
    /** \brief Holds parameters needed for BoundaryValuesFactor */
    std::map<std::string, boost::any> boundary_value_parameters;

    /** \brief FE vector holding solution from previous timestep */
    LA::MPI::Vector past_solution;
    /** \brief FE vector holding current solution iteration */
    LA::MPI::Vector current_solution;
    /** \brief Update vector for Newton-Rhapson method */
    LA::MPI::Vector system_rhs;


    /** \brief Which field theory to use -- LdG vs MS */
    std::string field_theory;

    double L2;
    double L3;

    /** \brief Alpha constant for bulk energy for the Maier-Saupe field theory*/
    double maier_saupe_alpha;

    /** \brief Object which handles Lagrange Multiplier inversion of Q-tensor */
    /** DIMENSIONALLY-DEPENDENT actually works fine for 3D but should make more efficient for 2D */
    LagrangeMultiplierAnalytic<dim> lagrange_multiplier;

    /** \brief constants for bulk energy for Landau-de Gennes field theory */
    double A;
    double B;
    double C;

    /** \brief Function which is evaluated at boundary to give Dirichlet vals */
    /** DIMENSIONALLY-DEPENDENT would need some work to make these independent */
    std::map<dealii::types::boundary_id, std::unique_ptr<BoundaryValues<dim>>> boundary_value_funcs;
    std::unique_ptr<BoundaryValues<dim>> initial_value_func;
    std::unique_ptr<BoundaryValues<dim>> left_internal_boundary_func;
    std::unique_ptr<BoundaryValues<dim>> right_internal_boundary_func;

    /** \brief vector holding t and spatial coordinates of defect points */
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
        // ar & boundary_value_funcs;
        // ar & lagrange_multiplier;

        // ar & field_theory;
        // ar & maier_saupe_alpha;
        // ar & L2;
        // ar & L3;

        // ar & A;
        // ar & B;
        // ar & C;

        // ar & defect_pts;
        // ar & energy_vals;
    }
};

#endif
