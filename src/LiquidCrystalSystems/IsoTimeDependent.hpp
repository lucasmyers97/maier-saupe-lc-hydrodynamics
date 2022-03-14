#ifndef ISO_TIME_DEPENDENT_HPP
#define ISO_TIME_DEPENDENT_HPP

#include <boost/program_options.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/vector.h>

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
class IsoTimeDependent
{
public:
    /**
     * \brief Default constructor, uses standard Lagrange elements and
     * initializes LagrangeMultiplier object to `alpha=1.0`, `tol=1e-8`,
     * `max_iters=10`.
     */
    IsoTimeDependent();

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
    IsoTimeDependent(const boost::program_options::variables_map &vm);

    /**
     * \brief Runs entire simulation, start to finish
     */
    void run();
    dealii::Functions::FEFieldFunction<dim> return_fe_field();

    /**
     * \brief Writes finite element grid to .h5 file according to grid found
     * in `grid_filename`, which are labeled by `meshgrid_names`.
     *
     * @param[in] grid_filename Name of `h5` file that holds the grid you are
     *            writing to.
     * @param[in] output_filename Name of file which will hold outputted grid
     *            and FE object values evaluated at gridpoints
     * @param[in] meshgrid_names Name of arrays which hold x, y, and/or z values
     *            of gridpoints in `grid_filename`.
     * @param[in] dist_scale Number which specifies ratio of meshgrid
     *            length-scale to finite element object length-scale.
     *            e.g. if meshgrid scale is 1km and FE scale is 1m, then
     *            dist_scale is 1000.
     */
    void write_to_grid(const std::string grid_filename,
                       const std::string output_filename,
                       const std::vector<std::string> meshgrid_names,
                       const double dist_scale) const;

    /**
     * \brief Reads function evaluated at gridpoints and creates finite element
     * object whose dof values are stored in `external_solution`.
     *
     * @param[in] system_filename Name of file holding function evaluated at
     *            gridpoints.
     * @param[in] dist_scale Number which specifies ratio of meshgrid
     *            length-scale to finite element object length-scale.
     *            See write_to_grid for details.
     */
    void read_from_grid(const std::string system_filename,
                        const double dist_scale);

    /**
     * \brief Creates a dim-dimensional hypercube and refines it.
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
     * \brief Outputs finite element object to vtu files
     *
     * @param[in] data_folder Path to folder where data is held.
     * @param[in] filename Name of data file being written
     */
    void output_results(const std::string data_folder,
                        const std::string filename,
                        const int timestep) const;

    /**
     * \brief Calculates differences in right-hand-side terms between
     * current_solution and external_solution.
     * Used for comparing solution to previously calculated solutions.
     */
    void calc_rhs_diff();

    /**
     * \brief Outputs previously calculated right-hand-side differences to
     * .vtu file.
     *
     * @param[in] filename Name of file that will hold the .vtu data.
     */
    void output_rhs_diff(const std::string filename);

  private:
    /**
     * \brief Solves for the system a time dt in the future, and puts the
     * current solution into the vector of past solutions
     */
    void iterate_timestep(const int current_timestep);
    /**
     * \brief Builds finite element matrix and right-hand side by iterating over
     * all active cells
     */
    void assemble_system(const int current_timestep);
    /** \brief Solves finite element linear system */
    void solve();
    /** \brief Sets Dirichlet boundary values on current_solution */
    void set_boundary_values();
    /** \brief determines Newton-Rhapson step-length (just 1.0 here) */
    double determine_step_length();

    /**
     * \brief Outputs picture of domain grid
     *
     * @param[in] data_folder Name of folder where file will go
     * @param[in] filename Name of file corresponding to grid picture
     */
    void output_grid(const std::string data_folder,
                     const std::string filename) const;
    /**
     * \brief Outputs a picture representing the matrix sparsity pattern
     *
     * @param[in] data_folder Name of folder where pattern will be outputted
     * @param[in] filename Name of file holding picture of sparsity pattern
     */
    void output_sparsity_pattern(const std::string data_folder,
                                 const std::string filename) const;

    // This gives boost::serialization access to private members
    friend class boost::serialization::access;

    /**
     * \brief Serializes class so that it can be read back into a different file
     */
    template <class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        ar & triangulation;
        ar & dof_handler;
        ar & fe;
        ar & system_matrix;

        ar & hanging_node_constraints;
        // ar & boundary_value_func;

        ar & current_solution;
        ar & system_update;
        ar & system_rhs;

        ar & lagrange_multiplier;

        ar & left_endpoint;
        ar & right_endpoint;
        ar & num_refines;

        ar & simulation_step_size;
        ar & simulation_tol;
        ar & simulation_max_iters;
        ar & maier_saupe_alpha;
        ar & boundary_values_name;
        ar & S_value;
        ar & defect_charge_name;

        ar & data_folder;
        ar & initial_config_filename;
        ar & final_config_filename;
        ar & archive_filename;
    }

    template <class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        ar & triangulation;
        ar & dof_handler;
        ar & fe;
        ar & system_matrix;

        ar & hanging_node_constraints;

        // dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
        // dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
        // hanging_node_constraints.condense(dsp);
        // sparsity_pattern.copy_from(dsp);

        // ar & boundary_value_func;

        ar & current_solution;
        ar & system_update;
        ar & system_rhs;

        ar & lagrange_multiplier;

        ar & left_endpoint;
        ar & right_endpoint;
        ar & num_refines;

        ar & simulation_step_size;
        ar & simulation_tol;
        ar & simulation_max_iters;
        ar & maier_saupe_alpha;
        ar & boundary_values_name;
        ar & S_value;
        ar & defect_charge_name;

        ar & data_folder;
        ar & initial_config_filename;
        ar & final_config_filename;
        ar & archive_filename;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** \brief Holds data associated with gridded domain */
    dealii::Triangulation <dim> triangulation;
    /** \brief Connects mesh and FE functions to problem vector and matrix */
    dealii::DoFHandler<dim> dof_handler;
    /** \brief Takes care of values associated with FE basis functions */
    dealii::FESystem<dim> fe;
    /** \brief Deals with FE matrix sparsity pattern */
    dealii::SparsityPattern sparsity_pattern;
    /** \brief Matrix for the linear FE problem */
    dealii::SparseMatrix<double> system_matrix;

    /** \brief Takes care of assigning boundary values to FE vector */
    dealii::AffineConstraints<double> hanging_node_constraints;
    /** \brief Function which is evaluated at boundary to give Dirichlet vals */
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    std::vector<dealii::Vector<double>> past_solutions;
    /** \brief FE vector holding current solution iteration */
    dealii::Vector<double> current_solution;
    /** \brief Update vector for Newton-Rhapson method */
    dealii::Vector<double> system_update;
    /** \brief FE system right-hand side for current iteration */
    dealii::Vector<double> system_rhs;

    /** \brief Optional vector holding external solution that is to be compared
     * against */
    dealii::Vector<double> external_solution;
    /** \brief Difference in bulk energy right-hand side terms generated by
     *  current and external solutions */
    dealii::Vector<double> rhs_bulk_term;
    /** \brief Difference in elastic energy right-hand side terms generated by
     *  current and external solutions */
    dealii::Vector<double> rhs_elastic_term;
    /** \brief Difference in Lagrange Multiplier energy right-hand side terms
     *  generated by current and external solutions */
    dealii::Vector<double> rhs_lagrange_term;

    /** \brief Object which handles Lagrange Multiplier inversion of Q-tensor */
    LagrangeMultiplier<order> lagrange_multiplier;

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
    /** \brief Time-step of simulation */
    double dt;
    /** \brief Number of total time-steps for simulation */
    int n_steps;

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

    /** \brief Fudge factor for resizing domain of external solution --
     *  necessary so that points where the external solution is read off fits
     *  into the domain of the finite element problem */
    double fudge_factor = 1.0001;
};

#endif
