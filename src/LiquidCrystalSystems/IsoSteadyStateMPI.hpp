#ifndef ISO_STEADY_STATE_MPI_HPP
#define ISO_STEADY_STATE_MPI_HPP

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/timer.h>

namespace LA = dealii::LinearAlgebraPETSc;

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/sparsity_tools.h>

#include <boost/program_options.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>

#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/DefectConfiguration.hpp"
#include "BoundaryValues/UniformConfiguration.hpp"
#include "Numerics/LagrangeMultiplier.hpp"

#include <memory>
#include <string>

/**
 * \brief Runs a simulation of a liquid crystal system using a Maier-Saupe free
 * energy model for the bulk energy, and a Frank elastic free energy.
 * Solves for the lowest energy (steady state) solution of the system.
 * Does this in a parallel distributed framework using MPI, so that it can scale
 * to much larger systems.
 *
 * See maier-saupe-weak-form.pdf for details, but the equation that this
 * class solves using the deal.II finite element package is given by:
 * \f{equation}{
 * 0 = \alpha Q_i +  \partial_k^2 Q_i - \Lambda_i
 * \f}
 * For now this is done on a hypercube grid (either square or cube for 2D or 3D
 * respectively).
 * It is solved using an iterative Newton-Rhapson method.
 */
template <int dim, int order>
class IsoSteadyStateMPI
{
public:
    /**
     * \brief Constructor that initializes using `variables_map` object from
     * Boost::program_options.
     * This just reads input arguments from the command line.
     *
     * @param[in] vm Map object which holds key-value pairs from the
     * command-line.
     * The input variables are as follows:
     *
     * `lagrange-step-size` -- step size in LagrangeMultiplier Newton-Rhapson
     * scheme
     *
     * `lagrange-tol` -- tolerance in LagrangeMultiplier Newton-Rhapson scheme
     *
     * `lagrange-max-iters` -- maximum number of iterations in
     * LagrangeMultiplier Newton-Rhapson scheme
     *
     * `left-endpoint` -- left endpoint of hypercube domain
     *
     * `right-endpoint` -- right endpoint of hypercube domain
     *
     * `num-refines` -- number of domain refines on hypercube
     *
     * `simulation-step-size` -- step size of finite element Newton-Rhapson
     * scheme
     *
     * `simulation-tol` -- tolerance of finite element Newton-Rhapson scheme
     *
     * `simulation-max-iters` -- maximum number of iterations for finite element
     * Newton-Rhapson scheme
     *
     * `maier-saupe-alpha` -- nondimensionalized alpha constant in Maier-Saupe
     * free energy
     *
     * `boundary-values-name` -- name of boundary value class for this problem.
     * So far we have "uniform", and "defect"
     *
     * `S-value` -- S value of starting configuration at the boundaries. This
     * is also the fixed S-value at the boundary in the case of Dirichlet
     * conditions.
     *
     * `defect-charge-name` -- if `boundary-values-name` is "defect", then this
     * is the name of the associated charge. Can be "plus-half", "minus-half",
     * "plus-one", or "minus-one" at the current time.
     *
     * `data-folder` -- folder that the data will be written to, relative to the
     * directory calling the program.
     *
     * `initial-config-filename` -- filename of the vtu file containing the
     * initial configuration.
     *
     * `final-config-filename` -- filename of the vtu file containing the final
     * configuration.
     *
     * `archive-filename` -- filename of the .dat archive file containing all of
     * the data necessary to reconstruct an instance of this class.
     */
    IsoSteadyStateMPI(const boost::program_options::variables_map &vm);

    /**
     * \brief Runs entire simulation, start to finish
     */
    void run();


  private:
    /**
     * \brief Creates a dim-dimensional hypercube in parallel and refines it.
     *
     * @param[in] num_refines Number of hypercube mesh refinements
     * @param[in] left Left endpoint of hypercube
     * @param[in] right Right endpoint of hypercube
     */
    void make_grid(const unsigned int num_refines, const double left = 0,
                   const double right = 1);
    /**
     * \brief On first step it initializes the finite-element object values,
     * and afterwards reinitializes relevant vectors/matrices with relevant
     * sparsity patterns.
     *
     * @param[in] initial_step Whether it is initial step in Newton-Rhapson
     *            scheme.
     */
    void setup_system(bool initial_step);
    /**
     * \brief Builds finite element matrix and right-hand side by iterating over
     * all active cells
     */
    void assemble_system(int step);
    /** \brief Solves finite element linear system */
    void solve();
    /** \brief Sets Dirichlet boundary values on current_solution */
    double determine_step_length();
    /**
     * \brief Outputs finite element object to vtu files
     *
     * @param[in] data_folder Path to folder where data is held.
     * @param[in] filename Name of data file being written
     */
    void output_results(const std::string data_folder,
                        const std::string filename,
                        const int step) const;

    /** \brief Controls mpi communication */
    MPI_Comm mpi_communicator;
    /** \brief Rank of current mpi process */
    int rank;
    /** \brief Total number of mpi ranks for simulation*/
    int num_ranks;
    /** \brief Parallel domain triangulation -- created by p4est */
    dealii::parallel::distributed::Triangulation<dim> triangulation;
    /** \brief Holds data associated with gridded domain */
    dealii::DoFHandler<dim> dof_handler;
    /** \brief Takes care of values associated with FE basis functions */
    dealii::FESystem<dim> fe;

    /** \brief Dofs owned by current mpi rank */
    dealii::IndexSet locally_owned_dofs;
    /** \brief Dofs relevant to current mpi rank (owned dofs + ghosted dofs) */
    dealii::IndexSet locally_relevant_dofs;

    /** \brief Takes care of assigning boundary values to FE vector */
    dealii::AffineConstraints<double> constraints;

    /** \brief Matrix for the linear FE problem -- distributed */
    LA::MPI::SparseMatrix system_matrix;
    /** \brief Right-hand side for linear FE problem -- distributed*/
    LA::MPI::Vector system_rhs;
    /** \brief ghosted vector holding all relevant solution entries */
    LA::MPI::Vector locally_relevant_solution;

    /** \brief Like cout, except only outputs to console if it is mpi rank 0 */
    dealii::ConditionalOStream pcout;
    /** \brief Handles outputting timing of all functions */
    dealii::TimerOutput computing_timer;

    /** \brief Object which handles Lagrange Multiplier inversion of Q-tensor */
    LagrangeMultiplier<order> lagrange_multiplier;

    /** \brief Function which is evaluated at boundary to give Dirichlet vals */
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    /** \brief Left endpoint of hypercube domain */
    double left_endpoint;
    /** \brief Right endpoint of hypercube domain */
    double right_endpoint;
    /** \brief Number of refines of hypercube domain */
    double num_refines;

    /** \brief Step size of Newton-Rhapson method for FE system */
    double simulation_step_size;
    /** \brief Error tolerance of normed residual for NR method for system */
    double simulation_tol;
    /** \brief Maximum number of iterations of system Newton-Rhapson method */
    int simulation_max_iters;
    /** \brief Alpha constant for bulk energy for the Maier-Saupe field theory*/
    double maier_saupe_alpha;

    /** \brief Name corresponding to boundary value (e.g. uniform, defect) */
    std::string boundary_values_name;
    /** \brief Initial S-value at the boundary of the domain */
    double S_value;
    /** \brief Name of defect charge for defect boundary condition (e.g.
     *  plus-half, minus-half, plus-one, minus-one) */
    std::string defect_charge_name;

    /** \brief Folder where data is written to */
    std::string data_folder;
    /** \brief Filename of initial configuration .vtu file */
    std::string initial_config_filename;
    /** \brief Filename of final configuration .vtu file */
    std::string final_config_filename;
    /** \brief Filename of serialized class data */
    std::string archive_filename;
};

#endif
