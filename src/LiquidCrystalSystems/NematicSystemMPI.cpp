#include "NematicSystemMPI.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/hdf5.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/differentiation/ad/ad_helpers.h>

#include <boost/program_options.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/any.hpp>

#include "Numerics/SetDefectBoundaryConstraints.hpp"
#include "Utilities/Output.hpp"
#include "Utilities/maier_saupe_constants.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Postprocessors/DirectorPostprocessor.hpp"
#include "Postprocessors/SValuePostprocessor.hpp"
#include "Postprocessors/EvaluateFEObject.hpp"
#include "Postprocessors/NematicPostprocessor.hpp"
#include "Postprocessors/EnergyPostprocessor.hpp"
#include "Postprocessors/ConfigurationForcePostprocessor.hpp"
#include "Numerics/FindDefects.hpp"
#include "nematic_assembly/nematic_assembly.hpp"
#include "nematic_energy/nematic_energy.hpp"

#include <deal.II/numerics/vector_tools_boundary.h>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <utility>
#include <tuple>



template <int dim>
NematicSystemMPI<dim>::
NematicSystemMPI(const dealii::parallel::distributed::Triangulation<dim>
                 &triangulation,
                 unsigned int degree,
                 std::string boundary_values_name,
                 const std::map<std::string, boost::any> &am,
                 double maier_saupe_alpha_,
                 double L2_,
                 double L3_,
                 double A_,
                 double B_,
                 double C_,
                 std::string field_theory_,
                 int order,
                 double lagrange_step_size,
                 double lagrange_tol,
                 unsigned int lagrange_max_iters)
    : dof_handler(triangulation)
    , fe(dealii::FE_Q<dim>(degree), maier_saupe_constants::vec_dim<dim>)
    , boundary_value_parameters(am)
    , boundary_value_func(BoundaryValuesFactory::
                          BoundaryValuesFactory<dim>(am))
    , lagrange_multiplier(order,
                          lagrange_step_size,
                          lagrange_tol,
                          lagrange_max_iters)

    , maier_saupe_alpha(maier_saupe_alpha_)
    , L2(L2_)
    , L3(L3_)
    , A(A_)
    , B(B_)
    , C(C_)
    , field_theory(field_theory_)

    , defect_pts(/* time + dim + charge = */ dim + 2) /** DIMENSIONALLY-DEPENDENT */
    , energy_vals(/* time + number of energy terms + squared energy = */ 6)
{}



template <int dim>
void NematicSystemMPI<dim>::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Nematic system MPI");

    prm.enter_subsection("Field theory");
    prm.declare_entry("Field theory",
                      "MS",
                      dealii::Patterns::Selection("MS|LdG"),
                      "Field theory to use for evolution of Nematic; "
                      "Maier-saupe or Landau-de Gennes");
    prm.declare_entry("L2",
                      "0.0",
                      dealii::Patterns::Double(),
                      "L2 elastic parameter");
    prm.declare_entry("L3",
                      "0.0",
                      dealii::Patterns::Double(),
                      "L3 elastic parameter");

    prm.enter_subsection("Maier saupe");
    prm.declare_entry("Maier saupe alpha",
                      "8.0",
                      dealii::Patterns::Double(),
                      "Alpha for Maier-saupe field theory -- "
                      "the alignment parameter");
    prm.declare_entry("Lebedev order",
                      "590",
                      dealii::Patterns::Integer(),
                      "Order of Lebedev quadrature when calculating "
                      "spherical integrals to invert singular potential");
    prm.declare_entry("Lagrange step size",
                      "1.0",
                      dealii::Patterns::Double(),
                      "Newton step size for inverting singular potential");
    prm.declare_entry("Lagrange tolerance",
                      "1e-10",
                      dealii::Patterns::Double(),
                      "Max L2 norm of residual from Newton's method when "
                      "calculating singular potential");
    prm.declare_entry("Lagrange maximum iterations",
                      "20",
                      dealii::Patterns::Integer(),
                      "Maximum number of Newton iterations when calculating "
                      "singular potential");
    prm.leave_subsection();

    prm.enter_subsection("Landau-de gennes");
    prm.declare_entry("A",
                      "-0.064",
                      dealii::Patterns::Double(),
                      "A parameter value for Landau-de Gennes potential");
    prm.declare_entry("B",
                      "-1.57",
                      dealii::Patterns::Double(),
                      "B parameter value for Landau-de Gennes potential");
    prm.declare_entry("C",
                      "1.29",
                      dealii::Patterns::Double(),
                      "C parameter value for Landau-de Gennes potential");
    prm.leave_subsection();

    prm.leave_subsection();

    BoundaryValuesFactory::declare_parameters<dim>(prm);

    prm.enter_subsection("Initial values");
        BoundaryValuesFactory::declare_parameters<dim>(prm);
    prm.leave_subsection();

    prm.enter_subsection("Internal boundary values");
        prm.enter_subsection("Left");
            BoundaryValuesFactory::declare_parameters<dim>(prm);
        prm.leave_subsection();
        prm.enter_subsection("Right");
            BoundaryValuesFactory::declare_parameters<dim>(prm);
        prm.leave_subsection();
    prm.leave_subsection();

    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPI<dim>::get_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("Nematic system MPI");

    prm.enter_subsection("Field theory");
    field_theory = prm.get("Field theory");
    L2 = prm.get_double("L2");
    L3 = prm.get_double("L3");

    prm.enter_subsection("Maier saupe");
    maier_saupe_alpha = prm.get_double("Maier saupe alpha");
    int order = prm.get_integer("Lebedev order");
    double lagrange_step_size = prm.get_double("Lagrange step size");
    double lagrange_tol = prm.get_double("Lagrange tolerance");
    int lagrange_max_iter = prm.get_integer("Lagrange maximum iterations");

    lagrange_multiplier = LagrangeMultiplierAnalytic<dim>(order, 
                                                          lagrange_step_size, 
                                                          lagrange_tol, 
                                                          lagrange_max_iter);
    prm.leave_subsection();

    prm.enter_subsection("Landau-de gennes");
    A = prm.get_double("A");
    B = prm.get_double("B");
    C = prm.get_double("C");
    prm.leave_subsection();

    prm.leave_subsection();

    boundary_value_parameters 
        = BoundaryValuesFactory::get_parameters<dim>(prm);
    boundary_value_func = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(boundary_value_parameters);

    prm.enter_subsection("Initial values");
    auto initial_value_parameters
        = BoundaryValuesFactory::get_parameters<dim>(prm);
    initial_value_func = BoundaryValuesFactory::
        BoundaryValuesFactory<dim>(initial_value_parameters);
    prm.leave_subsection();

    prm.enter_subsection("Internal boundary values");
        prm.enter_subsection("Left");
            auto left_internal_boundary_values
                = BoundaryValuesFactory::get_parameters<dim>(prm);
            left_internal_boundary_func = BoundaryValuesFactory::
                BoundaryValuesFactory<dim>(left_internal_boundary_values);
        prm.leave_subsection();
        prm.enter_subsection("Right");
            auto right_internal_boundary_values
                = BoundaryValuesFactory::get_parameters<dim>(prm);
            right_internal_boundary_func = BoundaryValuesFactory::
                BoundaryValuesFactory<dim>(right_internal_boundary_values);
        prm.leave_subsection();
    prm.leave_subsection();

    prm.leave_subsection();
}



template <int dim>
void NematicSystemMPI<dim>::
setup_dofs(const MPI_Comm &mpi_communicator, const bool initial_step)
{
    if (initial_step)
    {
        dof_handler.distribute_dofs(fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                        locally_relevant_dofs);

        // make constraints for system update
        constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                        constraints);

        if (boundary_value_func->return_boundary_condition() == std::string("Dirichlet"))
            dealii::VectorTools::
                interpolate_boundary_values(dof_handler,
                                            /* boundary_component = */0,
                                            dealii::Functions::
                                            ZeroFunction<dim>(fe.n_components()),
                                            constraints);

        /** DIMENSIONALLY-DEPENDENT this whole block */
        {
            std::vector<dealii::Point<dim>> 
                domain_defect_pts = boundary_value_func->return_defect_pts();
            const std::size_t n_defects = domain_defect_pts.size();
            std::map<dealii::types::material_id, const dealii::Function<dim>*>
                function_map;

            dealii::Functions::ZeroFunction<dim> 
                homogeneous_dirichlet_function(fe.n_components());
            for (dealii::types::material_id i = 1; i <= n_defects; ++i)
                function_map[i] = &homogeneous_dirichlet_function;

            std::map<dealii::types::global_dof_index, double> boundary_values;

            SetDefectBoundaryConstraints::
                interpolate_boundary_values(dof_handler, 
                                            function_map, 
                                            boundary_values);

            for (const auto &boundary_value : boundary_values)
            {
                if (constraints.can_store_line(boundary_value.first) &&
                    !constraints.is_constrained(boundary_value.first))
                {
                  constraints.add_line(boundary_value.first);
                  constraints.set_inhomogeneity(boundary_value.first,
                                                boundary_value.second);
                }
            }
        }
        constraints.make_consistent_in_parallel(locally_owned_dofs, 
                                                locally_relevant_dofs, 
                                                mpi_communicator);
        constraints.close();
    }
    // make sparsity pattern based on constraints
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                            dsp,
                                            constraints,
                                            /*keep_constrained_dofs=*/false);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp,
                                                       locally_owned_dofs,
                                                       mpi_communicator,
                                                       locally_relevant_dofs);
    constraints.condense(dsp);

    system_rhs.reinit(locally_owned_dofs,
                      mpi_communicator);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    system_matrix.compress(dealii::VectorOperation::insert);
    system_rhs.compress(dealii::VectorOperation::insert);
}



template <int dim>
void NematicSystemMPI<dim>::
initialize_fe_field(const MPI_Comm &mpi_communicator)
{
    // impose boundary conditions on initial condition
    dealii::AffineConstraints<double> configuration_constraints;
    configuration_constraints.clear();
    configuration_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::
        make_hanging_node_constraints(dof_handler,
                                      configuration_constraints);

    if (boundary_value_func->return_boundary_condition() == std::string("Dirichlet"))
        dealii::VectorTools::
            interpolate_boundary_values(dof_handler,
                                        /* boundary_component = */0,
                                        *boundary_value_func,
                                        configuration_constraints);
    /** DIMENSIONALLY-DEPENDENT this chunk */
    {
        std::map<dealii::types::material_id, const dealii::Function<dim>*>
            function_map;

        function_map[1] = left_internal_boundary_func.get();
        function_map[2] = right_internal_boundary_func.get();

        std::map<dealii::types::global_dof_index, double> boundary_values;

        SetDefectBoundaryConstraints::
            interpolate_boundary_values(dof_handler, 
                                        function_map, 
                                        boundary_values);

        for (const auto &boundary_value : boundary_values)
            if (configuration_constraints.can_store_line(boundary_value.first) &&
                !configuration_constraints.is_constrained(boundary_value.first))
            {
              configuration_constraints.add_line(boundary_value.first);
              configuration_constraints.set_inhomogeneity(boundary_value.first,
                                                          boundary_value.second);
            }
    }
    configuration_constraints.make_consistent_in_parallel(locally_owned_dofs, 
                                                          locally_relevant_dofs, 
                                                          mpi_communicator);
    configuration_constraints.close();

    // interpolate initial condition
    LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                           mpi_communicator);
    dealii::VectorTools::interpolate(dof_handler,
                                     *initial_value_func,
                                     locally_owned_solution);
    configuration_constraints.distribute(locally_owned_solution);
    locally_owned_solution.compress(dealii::VectorOperation::insert);

    // write completely distributed solution to current and past solutions
    current_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    past_solution.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);
    current_solution = locally_owned_solution;
    past_solution = locally_owned_solution;

    current_solution.compress(dealii::VectorOperation::insert);
    past_solution.compress(dealii::VectorOperation::insert);
}



template <int dim>
void NematicSystemMPI<dim>::
initialize_fe_field(const MPI_Comm &mpi_communicator,
                    LA::MPI::Vector &locally_owned_solution)
{
    // impose boundary conditions on initial condition
    dealii::AffineConstraints<double> configuration_constraints;
    configuration_constraints.clear();
    configuration_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::
        make_hanging_node_constraints(dof_handler,
                                      configuration_constraints);
    dealii::VectorTools::
        interpolate_boundary_values(dof_handler,
                                    /* boundary_component = */0,
                                    *boundary_value_func,
                                    configuration_constraints);
    configuration_constraints.close();

    // interpolate boundary values for inputted solution
    configuration_constraints.distribute(locally_owned_solution);
    locally_owned_solution.compress(dealii::VectorOperation::insert);

    // write completely distributed solution to current and past solutions
    current_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    past_solution.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);
    current_solution = locally_owned_solution;
    past_solution = locally_owned_solution;

    current_solution.compress(dealii::VectorOperation::insert);
    past_solution.compress(dealii::VectorOperation::insert);
}



/** DIMENSIONALLY-DEPENDENT need to regenerate assembly code */
template <int dim>
void NematicSystemMPI<dim>::
assemble_system(double dt, double theta, std::string &time_discretization)
{
    if (field_theory == "MS" && time_discretization == "convex_splitting")
        nematic_assembly::singular_potential_convex_splitting(dt, maier_saupe_alpha, 
                                                              L2, L3, 
                                                              dof_handler, 
                                                              current_solution, 
                                                              past_solution, 
                                                              lagrange_multiplier, 
                                                              constraints, 
                                                              system_matrix, 
                                                              system_rhs);
    else if (field_theory == "MS" && time_discretization == "semi_implicit")
        nematic_assembly::singular_potential_semi_implicit(dt, theta, maier_saupe_alpha, 
                                                           L2, L3, 
                                                           dof_handler, 
                                                           current_solution, 
                                                           past_solution, 
                                                           lagrange_multiplier, 
                                                           constraints, 
                                                           system_matrix, 
                                                           system_rhs);
    else if (field_theory == "LdG" && time_discretization == "convex_splitting")
        nematic_assembly::landau_de_gennes_convex_splitting(dt, A, B, C,
                                                            L2, L3,
                                                            dof_handler,
                                                            current_solution,
                                                            past_solution,
                                                            constraints,
                                                            system_matrix,
                                                            system_rhs);
}



template <int dim>
void NematicSystemMPI<dim>::solve_and_update(const MPI_Comm &mpi_communicator,
                                             const double alpha)
{
    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-10);
    LA::SolverGMRES solver(solver_control);
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix);

    LA::MPI::Vector completely_distributed_update(locally_owned_dofs,
                                                  mpi_communicator);
    solver.solve(system_matrix,
                 completely_distributed_update,
                 system_rhs,
                 preconditioner);
    constraints.distribute(completely_distributed_update);

    // update current_solution -- must transfer to completely distributed vector
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    completely_distributed_solution = current_solution;
    completely_distributed_solution.add(alpha, completely_distributed_update);
    current_solution = completely_distributed_solution;
}



template <int dim>
void NematicSystemMPI<dim>::update_forward_euler(const MPI_Comm &mpi_communicator, double dt)
{
    dealii::SolverControl solver_control(dof_handler.n_dofs(), 1e-10);
    LA::SolverCG solver(solver_control);
    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix);

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs,
                 preconditioner);
    constraints.distribute(completely_distributed_solution);
    completely_distributed_solution.sadd(dt, current_solution);

    current_solution = completely_distributed_solution;
}



template <int dim>
double NematicSystemMPI<dim>::return_norm()
{
    return system_rhs.l2_norm();
}



template <int dim>
double NematicSystemMPI<dim>::return_linfty_norm()
{
    return system_rhs.linfty_norm();
}



template <int dim>
void NematicSystemMPI<dim>::
set_past_solution_to_current(const MPI_Comm &mpi_communicator)
{
    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);
    completely_distributed_solution = current_solution;
    past_solution = completely_distributed_solution;
}


/** DIMENSIONALLY-DEPENDENT no concept of point defects in 3D */
template <int dim>
std::vector<std::vector<double>> NematicSystemMPI<dim>::
find_defects(double min_dist, 
             double charge_threshold, 
             double current_time)
{
    std::vector<dealii::Point<dim>> local_minima;
    std::vector<double> defect_charges;
    std::tie(local_minima, defect_charges) 
        = NumericalTools::find_defects(dof_handler, 
                                       current_solution, 
                                       min_dist, 
                                       charge_threshold);
    for (const auto &pt : local_minima)
    {
        defect_pts[0].push_back(current_time);
        defect_pts[1].push_back(pt[0]);
        defect_pts[2].push_back(pt[1]);
        if (dim == 3)
            defect_pts[3].push_back(pt[2]);
    }

    for (const auto &charge : defect_charges)
        defect_pts[dim + 1].push_back(charge);

    // have to use vector of vectors instead of Points to use MPI functions
    std::vector<std::vector<double>> cur_defect_pts(local_minima.size(),
                                                    std::vector<double>(dim));
    for (std::size_t i = 0; i < local_minima.size(); ++i)
        for (int j = 0; j < dim; ++j)
            cur_defect_pts[i][j] = local_minima[i][j];

    return cur_defect_pts;
}



/** DIMENSIONALLY-DEPENDENT need to regenerate energy calculation code */
template <int dim>
void NematicSystemMPI<dim>::
calc_energy(const MPI_Comm &mpi_communicator, double current_time)
{
    nematic_energy::singular_potential_energy(mpi_communicator, 
                                              current_time,
                                              maier_saupe_alpha, L2, L3,
                                              dof_handler,
                                              current_solution,
                                              lagrange_multiplier,
                                              energy_vals);
}



/** DIMENSIONALLY-DEPENDENT no point-defects in 3D */
template <int dim>
void NematicSystemMPI<dim>::
output_defect_positions(const MPI_Comm &mpi_communicator,
                        const std::string data_folder,
                        const std::string filename)
{
    std::vector<std::string> datanames = {"t", "x", "y"};
    if (dim == 3)
        datanames.push_back("z");
    datanames.push_back("charge");

    Output::distributed_vector_to_hdf5(defect_pts, 
                                       datanames, 
                                       mpi_communicator, 
                                       data_folder + filename 
                                       + std::string(".h5"));
}



template <int dim>
void NematicSystemMPI<dim>::
output_configuration_energies(const MPI_Comm &mpi_communicator,
                              const std::string data_folder,
                              const std::string filename)
{
    std::vector<std::string> datanames = {"t", 
                                          "mean_field_term",
                                          "entropy_term",
                                          "L1_elastic_term",
                                          "L2_elastic_term",
                                          "L3_elastic_term"};

    Output::distributed_vector_to_hdf5(energy_vals, 
                                       datanames, 
                                       mpi_communicator, 
                                       data_folder + filename 
                                       + std::string(".h5"));
}



template <int dim>
void NematicSystemMPI<dim>::
output_results(const MPI_Comm &mpi_communicator,
               const dealii::parallel::distributed::Triangulation<dim>
               &triangulation,
               const std::string folder,
               const std::string filename,
               const int time_step) const
{
    NematicPostprocessor<dim> nematic_postprocessor;
    EnergyPostprocessor<dim> energy_postprocessor(lagrange_multiplier, 
                                                  maier_saupe_alpha, 
                                                  L2, 
                                                  L3);
    ConfigurationForcePostprocessor<dim> 
        configuration_force_postprocessor(lagrange_multiplier, 
                                          maier_saupe_alpha, 
                                          L2, 
                                          L3);
    dealii::DataOut<dim> data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, nematic_postprocessor);
    data_out.add_data_vector(current_solution, energy_postprocessor);
    data_out.add_data_vector(current_solution, configuration_force_postprocessor);
    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    std::ofstream output(folder + filename
                         + "_" + std::to_string(time_step)
                         + ".vtu");
    data_out.write_vtu_with_pvtu_record(folder, filename, time_step,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



template <int dim>
void NematicSystemMPI<dim>::
output_Q_components(const MPI_Comm &mpi_communicator,
                    const dealii::parallel::distributed::Triangulation<dim>
                    &triangulation,
                    const std::string folder,
                    const std::string filename,
                    const int time_step) const
{
    dealii::DataOut<dim> data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> Q_names(fe.n_components());
    for (std::size_t i = 0; i < Q_names.size(); ++i)
        Q_names[i] = std::string("Q") + std::to_string(i);

    data_out.add_data_vector(current_solution, Q_names);
    dealii::Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    std::ofstream output(folder + filename + "_components"
                         + "_" + std::to_string(time_step)
                         + ".vtu");
    data_out.write_vtu_with_pvtu_record(folder, filename, time_step,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



template <int dim>
const dealii::DoFHandler<dim> &
NematicSystemMPI<dim>::return_dof_handler() const
{
    return dof_handler;
}



template <int dim>
const LA::MPI::Vector &
NematicSystemMPI<dim>::return_current_solution() const
{
    return current_solution;
}



template <int dim>
const dealii::AffineConstraints<double>&
NematicSystemMPI<dim>::return_constraints() const
{
    return constraints;
}



/** DIMENSIONALLY-DEPENDENT no point defects in 3D */
template <int dim>
std::vector<dealii::Point<dim>>
NematicSystemMPI<dim>::
return_defect_positions_at_time(const MPI_Comm &mpi_communicator,
                                double time) const
{
    // get indices where defect_pts times equal time
    std::vector<std::size_t> time_indices;
    auto time_begin = defect_pts[0].begin();
    auto time_end = defect_pts[0].end();
    auto time_iterator = time_begin;

    while (true)
    {
        time_iterator = std::find(time_iterator, time_end, time);
        if (time_iterator == time_end)
            break;
        time_indices.push_back(std::distance(time_begin, time_iterator));
        ++time_iterator;
    }

    // fill in points according to entries in defect_pts matching time
    std::vector<std::vector<double>> points(time_indices.size(), 
                                            std::vector<double>(dim));
    for (std::size_t i = 0; i < time_indices.size(); ++i)
        for (std::size_t j = 0; j < dim; ++j)
            points[i][j] = defect_pts[j + 1][time_indices[i]];

    std::vector<std::vector<std::vector<double>>> all_points
        = dealii::Utilities::MPI::all_gather(mpi_communicator, points);

    std::vector<dealii::Point<dim>> current_defect_points;
    for (const auto &local_points : all_points)
        for (const auto &local_point : local_points)
        {
            dealii::Point<dim> pt;
            for (std::size_t i = 0; i < local_point.size(); ++i)
                pt[i] = local_point[i];
            current_defect_points.push_back(pt);
        }

    return current_defect_points;
}



template <int dim>
double NematicSystemMPI<dim>::return_parameters() const
{
    return maier_saupe_alpha;
}



/** DIMENSIONALLY-DEPENDENT no point defects in 3D */
template <int dim>
const std::vector<dealii::Point<dim>>& NematicSystemMPI<dim>::
return_initial_defect_pts() const
{
    return boundary_value_func->return_defect_pts();
}


template <int dim>
void NematicSystemMPI<dim>::
set_current_solution(const MPI_Comm &mpi_communicator,
                     const LA::MPI::Vector &distributed_solution)
{
    current_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    current_solution = distributed_solution;
}

template class NematicSystemMPI<2>;
template class NematicSystemMPI<3>;
