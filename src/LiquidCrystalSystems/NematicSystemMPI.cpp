#include "NematicSystemMPI.hpp"

#include <algorithm>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/hdf5.h>

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_update_flags.h>
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
#include <deal.II/lac/vector_operation.h>
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
#include "Postprocessors/DebuggingL3TermPostprocessor.hpp"
#include "Postprocessors/DisclinationChargePostprocessor.hpp"
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
#include "Postprocessors/SingularPotentialPostprocessor.hpp"
#include "Numerics/FindDefects.hpp"
#include "nematic_assembly/nematic_assembly.hpp"
#include "nematic_energy/nematic_energy.hpp"

#include <deal.II/numerics/vector_tools_boundary.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <utility>
#include <tuple>
#include <set>



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

    , field_theory(field_theory_)
    , L2(L2_)
    , L3(L3_)
    , maier_saupe_alpha(maier_saupe_alpha_)
    , lagrange_multiplier(order,
                          lagrange_step_size,
                          lagrange_tol,
                          lagrange_max_iters)
    , A(A_)
    , B(B_)
    , C(C_)
    // , boundary_value_funcs{{0, std::move(BoundaryValuesFactory::BoundaryValuesFactory<dim>(am))}}

    , defect_pts(/* time + dim + charge = */ dim + 2) /** DIMENSIONALLY-DEPENDENT */
    , energy_vals(/* time + number of energy terms + squared energy = */ 6)
{}



template <int dim>
NematicSystemMPI<dim>::
NematicSystemMPI(unsigned int degree,
                 const std::string& field_theory,
                 double L2,
                 double L3,

                 double maier_saupe_alpha,

                 double S0,
                 double W1,
                 double W2,
                 double omega,

                 LagrangeMultiplierAnalytic<dim>&& lagrange_multiplier,

                 double A,
                 double B,
                 double C,

                 std::map<dealii::types::boundary_id, std::unique_ptr<BoundaryValues<dim>>> boundary_value_funcs,
                 std::unique_ptr<BoundaryValues<dim>> initial_value_func,
                 std::unique_ptr<BoundaryValues<dim>> left_internal_boundary_func,
                 std::unique_ptr<BoundaryValues<dim>> right_internal_boundary_func,
                 std::vector<dealii::types::boundary_id> surface_potential_ids)
    : fe(dealii::FE_Q<dim>(degree), maier_saupe_constants::vec_dim<dim>)

    , field_theory(field_theory)

    , L2(L2)
    , L3(L3)

    , maier_saupe_alpha(maier_saupe_alpha)

    , S0(S0)
    , W1(W1)
    , W2(W2)
    , omega(omega)

    , lagrange_multiplier(lagrange_multiplier)

    , A(A)
    , B(B)
    , C(C)

    , boundary_value_funcs(std::move(boundary_value_funcs))
    , initial_value_func(std::move(initial_value_func))
    , left_internal_boundary_func(std::move(left_internal_boundary_func))
    , right_internal_boundary_func(std::move(right_internal_boundary_func))
    , surface_potential_ids(std::move(surface_potential_ids))

    , defect_pts(/* time + dim + charge = */ dim + 2) /** DIMENSIONALLY-DEPENDENT */
    , energy_vals(/* time + number of energy terms + squared energy = */ 6)
{}



// TODO: find where this is referenced, delete everything
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



// TODO: find where this is referenced, delete everything
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
    // boundary_value_func = BoundaryValuesFactory::
    //     BoundaryValuesFactory<dim>(boundary_value_parameters);

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
reinit_dof_handler(const dealii::Triangulation<dim> &tria)
{
    dof_handler.reinit(tria);
}



template <int dim>
void NematicSystemMPI<dim>::
setup_dofs(const MPI_Comm &mpi_communicator, 
           const bool grid_modified,
           const std::vector<PeriodicBoundaries<dim>> &periodic_boundaries)
{
    if (grid_modified)
    {
        dof_handler.distribute_dofs(fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                        locally_relevant_dofs);

        // make constraints for system update
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                        constraints);

        for (auto const& [boundary_id, boundary_func] : boundary_value_funcs)
            if (boundary_func->return_boundary_condition() == std::string("Dirichlet"))
                dealii::VectorTools::
                    interpolate_boundary_values(dof_handler,
                                                boundary_id,
                                                dealii::Functions::
                                                ZeroFunction<dim>(fe.n_components()),
                                                constraints);

        for (const auto &periodic_boundary : periodic_boundaries)
            periodic_boundary.apply_to_constraints(dof_handler, constraints);

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
setup_dofs(const MPI_Comm &mpi_communicator, 
           dealii::Triangulation<dim> &tria,
           double fixed_defect_radius,
           const std::vector<PeriodicBoundaries<dim>> &periodic_boundaries)
{
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);

    // make constraints for system update
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                    constraints);

    for (auto const& [boundary_id, boundary_func] : boundary_value_funcs)
        if (boundary_func->return_boundary_condition() == std::string("Dirichlet"))
            dealii::VectorTools::
                interpolate_boundary_values(dof_handler,
                                            boundary_id,
                                            dealii::Functions::
                                            ZeroFunction<dim>(fe.n_components()),
                                            constraints);

    for (const auto &periodic_boundary : periodic_boundaries)
        periodic_boundary.apply_to_constraints(dof_handler, constraints);

    /** DIMENSIONALLY-WEIRD relies on projection into x-y plane */
    /** Fixes defects at fixed_defect_radius as determined by defect_pts held by initial_value_func */
    {
        std::vector<dealii::Point<dim>> 
            domain_defect_pts = initial_value_func->return_defect_pts();
        const std::size_t n_defects = domain_defect_pts.size();
        std::map<dealii::types::material_id, const dealii::Function<dim>*>
            function_map;

        std::vector<dealii::types::material_id> defect_ids;
        for (std::size_t i = 1; i <= n_defects; ++i)
            defect_ids.push_back(i);

        SetDefectBoundaryConstraints::mark_defect_domains(tria, 
                                                          domain_defect_pts, 
                                                          defect_ids, 
                                                          fixed_defect_radius);

        dealii::Functions::ZeroFunction<dim> 
            homogeneous_dirichlet_function(fe.n_components());
        for (dealii::types::material_id i = 1; i <= n_defects; ++i)
            function_map[i] = &homogeneous_dirichlet_function;

        SetDefectBoundaryConstraints::
            interpolate_boundary_values(mpi_communicator,
                                        dof_handler, 
                                        function_map, 
                                        constraints);
    }
    constraints.close();

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
initialize_fe_field(const MPI_Comm &mpi_communicator,
                    const std::vector<PeriodicBoundaries<dim>> &periodic_boundaries)
{
    // impose boundary conditions on initial condition
    dealii::AffineConstraints<double> configuration_constraints;
    configuration_constraints.clear();
    configuration_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::
        make_hanging_node_constraints(dof_handler,
                                      configuration_constraints);

    for (auto const& [boundary_id, boundary_func] : boundary_value_funcs)
        if (boundary_func->return_boundary_condition() == std::string("Dirichlet"))
            dealii::VectorTools::
                interpolate_boundary_values(dof_handler,
                                            boundary_id,
                                            *boundary_func,
                                            configuration_constraints);

    for (const auto &periodic_boundary : periodic_boundaries)
        periodic_boundary.apply_to_constraints(dof_handler, configuration_constraints);

    /* WARNING: DEPENDS ON PREVIOUSLY SETTING TRIANGULATION */
    // freeze defects if it's marked on the triangulation
    std::map<dealii::types::material_id, const dealii::Function<dim>*>
        function_map;

    function_map[1] = left_internal_boundary_func.get();
    function_map[2] = right_internal_boundary_func.get();

    SetDefectBoundaryConstraints::
        interpolate_boundary_values(mpi_communicator,
                                    dof_handler, 
                                    function_map, 
                                    configuration_constraints);
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
                    LA::MPI::Vector &locally_owned_solution,
                    const std::vector<PeriodicBoundaries<dim>> &periodic_boundaries)
{
    // impose boundary conditions on initial condition
    dealii::AffineConstraints<double> configuration_constraints;
    configuration_constraints.clear();
    configuration_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::
        make_hanging_node_constraints(dof_handler,
                                      configuration_constraints);
    for (auto const& [boundary_id, boundary_func] : boundary_value_funcs)
        if (boundary_func->return_boundary_condition() == std::string("Dirichlet"))
            dealii::VectorTools::
                interpolate_boundary_values(dof_handler,
                                            boundary_id,
                                            *boundary_func,
                                            configuration_constraints);

    for (const auto &periodic_boundary : periodic_boundaries)
        periodic_boundary.apply_to_constraints(dof_handler, configuration_constraints);

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
    else if (field_theory == "MS" && time_discretization == "semi_implicit_rotated")
        nematic_assembly::singular_potential_semi_implicit_rotated(dt, theta, maier_saupe_alpha, omega,
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
    else if (field_theory == "MS" && time_discretization == "newtons_method")
        nematic_assembly::singular_potential_newtons_method(maier_saupe_alpha, 
                                                            L2, L3, 
                                                            dof_handler, 
                                                            current_solution, 
                                                            past_solution, 
                                                            lagrange_multiplier, 
                                                            constraints, 
                                                            system_matrix, 
                                                            system_rhs);
    else
        throw std::invalid_argument("Inputted incorrect nematic assembly parameters");

}



template <int dim>
void NematicSystemMPI<dim>::
assemble_boundary_terms(double dt, 
                        double theta, 
                        std::string &time_discretization)
{
    if (surface_potential_ids.empty())
        return;

    const dealii::FESystem<dim> fe = dof_handler.get_fe();
    dealii::QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    // system_matrix = 0;
    // system_rhs = 0;

    dealii::FEFaceValues<dim> fe_values(fe,
                                        face_quadrature_formula,
                                        dealii::update_values
                                        | dealii::update_normal_vectors
                                        | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = face_quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<dealii::Vector<double>>
        Q_vec(n_q_points, dealii::Vector<double>(fe.components));
    std::vector<dealii::Vector<double>>
        Q0_vec(n_q_points, dealii::Vector<double>(fe.components));

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const auto is_surface_potential_id 
        = [&ids = this->surface_potential_ids](dealii::types::boundary_id boundary_id) 
        {
            return std::find(ids.begin(), ids.end(), boundary_id) != ids.end();
        };

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if ( !cell->is_locally_owned() )
            continue;

        cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);

        for (const auto &face : cell->face_iterators())
        {
            if (!face->at_boundary() || 
                !is_surface_potential_id(face->boundary_id()))
                continue;

            fe_values.reinit(cell, face);

            fe_values.get_function_values(current_solution, Q_vec);
            fe_values.get_function_values(past_solution, Q0_vec);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;

                    if (component_i == 0)
                        cell_rhs(i) += (
                            (-2*Q_vec[q][0] - Q_vec[q][3] + 2*Q0_vec[q][0] + Q0_vec[q][3])*fe_values.shape_value(i, q)
                            +
                            (2.0/3.0)*dt*(theta*(W1*(-S0*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][3] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][1]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][4] - fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(Q0_vec[q][0] + Q0_vec[q][3]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][2]) + 3*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][1] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][0]) - 3*Q0_vec[q][0]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][0]) - theta*(W1*(-S0*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][2]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][4]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q0_vec[q][0] + Q0_vec[q][3])) + 3*Q0_vec[q][0] + 3*Q0_vec[q][3]) + 2*W2*(Q0_vec[q][0] + Q0_vec[q][3])*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))) - (theta - 1)*(W1*(-S0*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][3] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][4] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][1]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][4] - fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(Q_vec[q][0] + Q_vec[q][3]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][2]) + 3*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][1] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][0]) - 3*Q_vec[q][0]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][0]) + (theta - 1)*(W1*(-S0*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][2]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][4]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q_vec[q][0] + Q_vec[q][3])) + 3*Q_vec[q][0] + 3*Q_vec[q][3]) + 2*W2*(Q_vec[q][0] + Q_vec[q][3])*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))))*fe_values.shape_value(i, q)
                            ) * fe_values.JxW(q);
                    else if (component_i == 1)
                        cell_rhs(i) += (
                            2*(-Q_vec[q][1] + Q0_vec[q][1])*fe_values.shape_value(i, q)
                            +
                            (2.0/3.0)*dt*(theta*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][1] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][0]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][4] - fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(Q0_vec[q][0] + Q0_vec[q][3]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][2]) + 3*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][3] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][1]) - 3*Q0_vec[q][1]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][1]) + theta*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][3]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][2] - fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(Q0_vec[q][0] + Q0_vec[q][3]) + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][4]) + 3*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][1]) - 3*Q0_vec[q][1]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][1]) - (theta - 1)*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][1] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][0]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][4] - fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(Q_vec[q][0] + Q_vec[q][3]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][2]) + 3*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][3] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][4] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][1]) - 3*Q_vec[q][1]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][1]) - (theta - 1)*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][4] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][3]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][2] - fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(Q_vec[q][0] + Q_vec[q][3]) + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][4]) + 3*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][1]) - 3*Q_vec[q][1]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][1]))*fe_values.shape_value(i, q)
                            ) * fe_values.JxW(q);
                    else if (component_i == 2)
                        cell_rhs(i) += (
                            2*(-Q_vec[q][2] + Q0_vec[q][2])*fe_values.shape_value(i, q)
                            +
                            (2.0/3.0)*dt*(theta*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][4]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q0_vec[q][0] + Q0_vec[q][3])) + 3*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][2]) - 3*Q0_vec[q][2]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][2]) + theta*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][1] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][0]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][3] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][1]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][4] - fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(Q0_vec[q][0] + Q0_vec[q][3]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q0_vec[q][2]) - 3*Q0_vec[q][2]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][2]) - (theta - 1)*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][4]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q_vec[q][0] + Q_vec[q][3])) + 3*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][2]) - 3*Q_vec[q][2]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][2]) - (theta - 1)*(W1*(-S0*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][1] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][0]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][3] + fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][4] + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][1]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][4] - fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(Q_vec[q][0] + Q_vec[q][3]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*Q_vec[q][2]) - 3*Q_vec[q][2]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][2]))*fe_values.shape_value(i, q)
                            ) * fe_values.JxW(q);
                    else if (component_i == 3)
                        cell_rhs(i) += (
                            (-Q_vec[q][0] - 2*Q_vec[q][3] + Q0_vec[q][0] + 2*Q0_vec[q][3])*fe_values.shape_value(i, q)
                            +
                            (2.0/3.0)*dt*(theta*(W1*(-S0*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][1]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][2] - fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(Q0_vec[q][0] + Q0_vec[q][3]) + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][4]) + 3*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][3]) - 3*Q0_vec[q][3]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][3]) - theta*(W1*(-S0*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][2]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][4]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q0_vec[q][0] + Q0_vec[q][3])) + 3*Q0_vec[q][0] + 3*Q0_vec[q][3]) + 2*W2*(Q0_vec[q][0] + Q0_vec[q][3])*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))) - (theta - 1)*(W1*(-S0*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][1]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][2] - fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(Q_vec[q][0] + Q_vec[q][3]) + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][4]) + 3*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][4] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][3]) - 3*Q_vec[q][3]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][3]) + (theta - 1)*(W1*(-S0*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][2]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][4]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q_vec[q][0] + Q_vec[q][3])) + 3*Q_vec[q][0] + 3*Q_vec[q][3]) + 2*W2*(Q_vec[q][0] + Q_vec[q][3])*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))))*fe_values.shape_value(i, q)
                            ) * fe_values.JxW(q);
                    else if (component_i == 4)
                        cell_rhs(i) += (
                            2*(-Q_vec[q][4] + Q0_vec[q][4])*fe_values.shape_value(i, q)
                            +
                            (2.0/3.0)*dt*(theta*(W1*(-S0*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][2]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q0_vec[q][0] + Q0_vec[q][3])) + 3*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q0_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q0_vec[q][4]) - 3*Q0_vec[q][4]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][4]) + theta*(W1*(-S0*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][2] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][1]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q0_vec[q][4] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][3]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q0_vec[q][2] - fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(Q0_vec[q][0] + Q0_vec[q][3]) + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q0_vec[q][4]) - 3*Q0_vec[q][4]) - 2*W2*(-2*(S0) * (S0) + 3*(Q0_vec[q][0] + Q0_vec[q][3]) * (Q0_vec[q][0] + Q0_vec[q][3]) + 3*(Q0_vec[q][0]) * (Q0_vec[q][0]) + 6*(Q0_vec[q][1]) * (Q0_vec[q][1]) + 6*(Q0_vec[q][2]) * (Q0_vec[q][2]) + 3*(Q0_vec[q][3]) * (Q0_vec[q][3]) + 6*(Q0_vec[q][4]) * (Q0_vec[q][4]))*Q0_vec[q][4]) - (theta - 1)*(W1*(-S0*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][2]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][4] - ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(Q_vec[q][0] + Q_vec[q][3])) + 3*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*Q_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][3] + ((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*Q_vec[q][4]) - 3*Q_vec[q][4]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][4]) - (theta - 1)*(W1*(-S0*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2] + 3*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][0] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][2] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][1]) + 3*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][1] + fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*Q_vec[q][4] + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][3]) + 3*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1)*(fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*Q_vec[q][2] - fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(Q_vec[q][0] + Q_vec[q][3]) + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*Q_vec[q][4]) - 3*Q_vec[q][4]) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4]))*Q_vec[q][4]))*fe_values.shape_value(i, q)
                            ) * fe_values.JxW(q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int component_j =
                            fe.system_to_component_index(j).first;

                        if (component_i == 0 && component_j == 0)
                            cell_matrix(i, j) += (
                                2*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                +
                                (2.0/3.0)*dt*(theta - 1)*(8*(S0) * (S0)*W2 + 3*W1*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 6*W1*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0])*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 6*W1*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) + 3*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 6*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 72*W2*(Q_vec[q][0]) * (Q_vec[q][0]) - 72*W2*Q_vec[q][0]*Q_vec[q][3] - 24*W2*(Q_vec[q][1]) * (Q_vec[q][1]) - 24*W2*(Q_vec[q][2]) * (Q_vec[q][2]) - 36*W2*(Q_vec[q][3]) * (Q_vec[q][3]) - 24*W2*(Q_vec[q][4]) * (Q_vec[q][4]))*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 0 && component_j == 1)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(-W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1) - 4*W2*(Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][1] - 4*W2*Q_vec[q][0]*Q_vec[q][1])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 0 && component_j == 2)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1) - W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 4*W2*(Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][2] - 4*W2*Q_vec[q][0]*Q_vec[q][2])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 0 && component_j == 3)
                            cell_matrix(i, j) += (
                                fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                +
                                (2.0/3.0)*dt*(theta - 1)*(4*(S0) * (S0)*W2 + 3*W1*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0])*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2])) - 3*W1*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1])*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 3*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 6*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 12*W2*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][0] - 24*W2*(Q_vec[q][0]) * (Q_vec[q][0]) - 48*W2*Q_vec[q][0]*Q_vec[q][3] - 12*W2*(Q_vec[q][1]) * (Q_vec[q][1]) - 12*W2*(Q_vec[q][2]) * (Q_vec[q][2]) - 36*W2*(Q_vec[q][3]) * (Q_vec[q][3]) - 12*W2*(Q_vec[q][4]) * (Q_vec[q][4]))*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 0 && component_j == 4)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0])*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2] - W1*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 4*W2*(Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][4] - 4*W2*Q_vec[q][0]*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 1 && component_j == 0)
                            cell_matrix(i, j) += (
                                -4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(-(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) + (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 1) + 4*W2*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][1])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 1 && component_j == 1)
                            cell_matrix(i, j) += (
                                2*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                +
                                (4.0/3.0)*dt*(theta - 1)*(3*W1*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0])*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1) - 1) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 18*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4])))*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 1 && component_j == 2)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(2*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1) - 8*W2*Q_vec[q][1]*Q_vec[q][2])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 1 && component_j == 3)
                            cell_matrix(i, j) += (
                                -4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(-(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) + (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 1) + 4*W2*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][1])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 1 && component_j == 4)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(2*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1) - 8*W2*Q_vec[q][1]*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 2 && component_j == 0)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2])) - 4*W2*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][2])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 2 && component_j == 1)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*(2*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1) - 8*W2*Q_vec[q][1]*Q_vec[q][2])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 2 && component_j == 2)
                            cell_matrix(i, j) += (
                                2*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                +
                                (4.0/3.0)*dt*(theta - 1)*(3*W1*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0])*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + ((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - 1)*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 1) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 18*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 6*(Q_vec[q][4]) * (Q_vec[q][4])))*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 2 && component_j == 3)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 1) - 4*W2*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][2])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 2 && component_j == 4)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(2*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 8*W2*Q_vec[q][2]*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 3 && component_j == 0)
                            cell_matrix(i, j) += (
                                fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                +
                                (2.0/3.0)*dt*(theta - 1)*(4*(S0) * (S0)*W2 - 3*W1*(fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0])*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 3*W1*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1])*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2])) + 3*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 6*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 12*W2*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][3] - 36*W2*(Q_vec[q][0]) * (Q_vec[q][0]) - 48*W2*Q_vec[q][0]*Q_vec[q][3] - 12*W2*(Q_vec[q][1]) * (Q_vec[q][1]) - 12*W2*(Q_vec[q][2]) * (Q_vec[q][2]) - 24*W2*(Q_vec[q][3]) * (Q_vec[q][3]) - 12*W2*(Q_vec[q][4]) * (Q_vec[q][4]))*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 3 && component_j == 1)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(-W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1) - 4*W2*(Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][1] - 4*W2*Q_vec[q][1]*Q_vec[q][3])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 3 && component_j == 2)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1])*fe_values.normal_vector(q)[2] - W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 4*W2*(Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][2] - 4*W2*Q_vec[q][2]*Q_vec[q][3])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 3 && component_j == 3)
                            cell_matrix(i, j) += (
                                2*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                +
                                (2.0/3.0)*dt*(theta - 1)*(8*(S0) * (S0)*W2 + 3*W1*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 6*W1*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1])*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 6*W1*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) + 3*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 6*W1*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 36*W2*(Q_vec[q][0]) * (Q_vec[q][0]) - 72*W2*Q_vec[q][0]*Q_vec[q][3] - 24*W2*(Q_vec[q][1]) * (Q_vec[q][1]) - 24*W2*(Q_vec[q][2]) * (Q_vec[q][2]) - 72*W2*(Q_vec[q][3]) * (Q_vec[q][3]) - 24*W2*(Q_vec[q][4]) * (Q_vec[q][4]))*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 3 && component_j == 4)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1) - W1*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 4*W2*(Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][4] - 4*W2*Q_vec[q][3]*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 4 && component_j == 0)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[0]) * (fe_values.normal_vector(q)[0]) - (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + 1) - 4*W2*(2*Q_vec[q][0] + Q_vec[q][3])*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 4 && component_j == 1)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[2]*(2*(fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1) - 8*W2*Q_vec[q][1]*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 4 && component_j == 2)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[0]*fe_values.normal_vector(q)[1]*(2*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 8*W2*Q_vec[q][2]*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 4 && component_j == 3)
                            cell_matrix(i, j) += (
                                4*dt*(theta - 1)*(W1*fe_values.normal_vector(q)[1]*fe_values.normal_vector(q)[2]*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - (fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2])) - 4*W2*(Q_vec[q][0] + 2*Q_vec[q][3])*Q_vec[q][4])*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);
                        else if (component_i == 4 && component_j == 4)
                            cell_matrix(i, j) += (
                                2*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                +
                                (4.0/3.0)*dt*(theta - 1)*(3*W1*((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1])*(fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) + ((fe_values.normal_vector(q)[1]) * (fe_values.normal_vector(q)[1]) - 1)*((fe_values.normal_vector(q)[2]) * (fe_values.normal_vector(q)[2]) - 1) - 1) - 2*W2*(-2*(S0) * (S0) + 3*(Q_vec[q][0] + Q_vec[q][3]) * (Q_vec[q][0] + Q_vec[q][3]) + 3*(Q_vec[q][0]) * (Q_vec[q][0]) + 6*(Q_vec[q][1]) * (Q_vec[q][1]) + 6*(Q_vec[q][2]) * (Q_vec[q][2]) + 3*(Q_vec[q][3]) * (Q_vec[q][3]) + 18*(Q_vec[q][4]) * (Q_vec[q][4])))*fe_values.shape_value(i, q)*fe_values.shape_value(j, q)
                                ) * fe_values.JxW(q);

                    }

                }
            }
        }
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
    }
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
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



template <>
std::vector<std::vector<double>> NematicSystemMPI<2>::
find_defects(double min_dist, 
             double charge_threshold, 
             double current_time)
{
    constexpr int dim = 2; 

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



template <>
std::vector<std::vector<double>> NematicSystemMPI<3>::
find_defects(double min_dist, 
             double charge_threshold, 
             double current_time)
{
    throw std::logic_error("find_defects not implemented in 3D");
}



/** DIMENSIONALLY-DEPENDENT need to regenerate energy calculation code */
template <int dim>
void NematicSystemMPI<dim>::
calc_energy(const MPI_Comm &mpi_communicator, double current_time, const std::string &time_discretization)
{
    if (field_theory == "MS" && time_discretization == "semi_implicit_rotated")
        nematic_energy::singular_potential_rot_energy(mpi_communicator, 
                                                      current_time,
                                                      maier_saupe_alpha, L2, L3, omega,
                                                      dof_handler,
                                                      current_solution,
                                                      lagrange_multiplier,
                                                      energy_vals);
    else
        nematic_energy::singular_potential_energy(mpi_communicator, 
                                                  current_time,
                                                  maier_saupe_alpha, L2, L3,
                                                  dof_handler,
                                                  current_solution,
                                                  lagrange_multiplier,
                                                  energy_vals);
}



template<int dim>
dealii::Vector<float> NematicSystemMPI<dim>::
calc_disclination_density()
{
    dealii::Vector<float> disclination_density(dof_handler.get_triangulation().n_active_cells());

    const dealii::FESystem<dim> fe = dof_handler.get_fe();
    dealii::QGauss<dim> quadrature_formula(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values
                                    | dealii::update_gradients
                                    | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<std::vector<dealii::Tensor<1, dim>>>
        dQ(n_q_points, std::vector<dealii::Tensor<1, dim, double>>(fe.components));

    auto cell = dof_handler.begin_active();
    auto endc = dof_handler.end();
    dealii::Vector<float>::size_type i = 0;

    for (; cell != endc; ++cell, ++i)
    {
        if ( !(cell->is_locally_owned()) )
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_gradients(current_solution, dQ);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            disclination_density[i] += (2*dQ[q][0][1]*dQ[q][4][2] 
                                        - 2*dQ[q][0][2]*dQ[q][4][1] 
                                        + 2*dQ[q][1][1]*dQ[q][2][2] 
                                        - 2*dQ[q][1][2]*dQ[q][2][1] 
                                        + 4*dQ[q][3][1]*dQ[q][4][2] 
                                        - 4*dQ[q][3][2]*dQ[q][4][1])
                                       * fe_values.JxW(q);
            // disclination_density[i] += std::sqrt(
            //     4*(dQ[q][0][0]*dQ[q][1][1] - dQ[q][0][1]*dQ[q][1][0] 
            //        + dQ[q][1][0]*dQ[q][3][1] - dQ[q][1][1]*dQ[q][3][0] 
            //        + dQ[q][2][0]*dQ[q][4][1] - dQ[q][2][1]*dQ[q][4][0]) 
            //       * (dQ[q][0][0]*dQ[q][1][1] - dQ[q][0][1]*dQ[q][1][0] 
            //         + dQ[q][1][0]*dQ[q][3][1] - dQ[q][1][1]*dQ[q][3][0] 
            //         + dQ[q][2][0]*dQ[q][4][1] - dQ[q][2][1]*dQ[q][4][0]) 
            //     + 4*(dQ[q][0][0]*dQ[q][1][2] - dQ[q][0][2]*dQ[q][1][0] 
            //          + dQ[q][1][0]*dQ[q][3][2] - dQ[q][1][2]*dQ[q][3][0] 
            //          + dQ[q][2][0]*dQ[q][4][2] - dQ[q][2][2]*dQ[q][4][0]) 
            //       * (dQ[q][0][0]*dQ[q][1][2] - dQ[q][0][2]*dQ[q][1][0] 
            //          + dQ[q][1][0]*dQ[q][3][2] - dQ[q][1][2]*dQ[q][3][0] 
            //          + dQ[q][2][0]*dQ[q][4][2] - dQ[q][2][2]*dQ[q][4][0]) 
            //     + 4*(2*dQ[q][0][0]*dQ[q][2][1] - 2*dQ[q][0][1]*dQ[q][2][0] 
            //          + dQ[q][1][0]*dQ[q][4][1] - dQ[q][1][1]*dQ[q][4][0] 
            //          - dQ[q][2][0]*dQ[q][3][1] + dQ[q][2][1]*dQ[q][3][0]) 
            //       * (2*dQ[q][0][0]*dQ[q][2][1] - 2*dQ[q][0][1]*dQ[q][2][0] 
            //          + dQ[q][1][0]*dQ[q][4][1] - dQ[q][1][1]*dQ[q][4][0] 
            //          - dQ[q][2][0]*dQ[q][3][1] + dQ[q][2][1]*dQ[q][3][0]) 
            //     + 4*(2*dQ[q][0][0]*dQ[q][2][2] - 2*dQ[q][0][2]*dQ[q][2][0] 
            //          + dQ[q][1][0]*dQ[q][4][2] - dQ[q][1][2]*dQ[q][4][0] 
            //          - dQ[q][2][0]*dQ[q][3][2] + dQ[q][2][2]*dQ[q][3][0]) 
            //       * (2*dQ[q][0][0]*dQ[q][2][2] - 2*dQ[q][0][2]*dQ[q][2][0] 
            //          + dQ[q][1][0]*dQ[q][4][2] - dQ[q][1][2]*dQ[q][4][0] 
            //          - dQ[q][2][0]*dQ[q][3][2] + dQ[q][2][2]*dQ[q][3][0]) 
            //     + 4*(dQ[q][0][0]*dQ[q][4][1] - dQ[q][0][1]*dQ[q][4][0] 
            //          + dQ[q][1][0]*dQ[q][2][1] - dQ[q][1][1]*dQ[q][2][0] 
            //          + 2*dQ[q][3][0]*dQ[q][4][1] - 2*dQ[q][3][1]*dQ[q][4][0]) 
            //       * (dQ[q][0][0]*dQ[q][4][1] - dQ[q][0][1]*dQ[q][4][0] 
            //          + dQ[q][1][0]*dQ[q][2][1] - dQ[q][1][1]*dQ[q][2][0] 
            //          + 2*dQ[q][3][0]*dQ[q][4][1] - 2*dQ[q][3][1]*dQ[q][4][0]) 
            //     + 4*(dQ[q][0][0]*dQ[q][4][2] - dQ[q][0][2]*dQ[q][4][0] 
            //          + dQ[q][1][0]*dQ[q][2][2] - dQ[q][1][2]*dQ[q][2][0] 
            //          + 2*dQ[q][3][0]*dQ[q][4][2] - 2*dQ[q][3][2]*dQ[q][4][0]) 
            //       * (dQ[q][0][0]*dQ[q][4][2] - dQ[q][0][2]*dQ[q][4][0] 
            //          + dQ[q][1][0]*dQ[q][2][2] - dQ[q][1][2]*dQ[q][2][0] 
            //          + 2*dQ[q][3][0]*dQ[q][4][2] - 2*dQ[q][3][2]*dQ[q][4][0]) 
            //     + 4*(dQ[q][0][1]*dQ[q][1][2] - dQ[q][0][2]*dQ[q][1][1] 
            //          + dQ[q][1][1]*dQ[q][3][2] - dQ[q][1][2]*dQ[q][3][1] 
            //          + dQ[q][2][1]*dQ[q][4][2] - dQ[q][2][2]*dQ[q][4][1]) 
            //       * (dQ[q][0][1]*dQ[q][1][2] - dQ[q][0][2]*dQ[q][1][1] 
            //          + dQ[q][1][1]*dQ[q][3][2] - dQ[q][1][2]*dQ[q][3][1] 
            //          + dQ[q][2][1]*dQ[q][4][2] - dQ[q][2][2]*dQ[q][4][1]) 
            //     + 4*(2*dQ[q][0][1]*dQ[q][2][2] - 2*dQ[q][0][2]*dQ[q][2][1] 
            //          + dQ[q][1][1]*dQ[q][4][2] - dQ[q][1][2]*dQ[q][4][1] 
            //          - dQ[q][2][1]*dQ[q][3][2] + dQ[q][2][2]*dQ[q][3][1]) 
            //       * (2*dQ[q][0][1]*dQ[q][2][2] - 2*dQ[q][0][2]*dQ[q][2][1] 
            //          + dQ[q][1][1]*dQ[q][4][2] - dQ[q][1][2]*dQ[q][4][1] 
            //          - dQ[q][2][1]*dQ[q][3][2] + dQ[q][2][2]*dQ[q][3][1]) 
            //     + 4*(dQ[q][0][1]*dQ[q][4][2] - dQ[q][0][2]*dQ[q][4][1] 
            //          + dQ[q][1][1]*dQ[q][2][2] - dQ[q][1][2]*dQ[q][2][1] 
            //          + 2*dQ[q][3][1]*dQ[q][4][2] - 2*dQ[q][3][2]*dQ[q][4][1]) 
            //       * (dQ[q][0][1]*dQ[q][4][2] - dQ[q][0][2]*dQ[q][4][1] 
            //          + dQ[q][1][1]*dQ[q][2][2] - dQ[q][1][2]*dQ[q][2][1] 
            //          + 2*dQ[q][3][1]*dQ[q][4][2] - 2*dQ[q][3][2]*dQ[q][4][1])
            //     ) * fe_values.JxW(q);
        }
    }

    return disclination_density;
}



/** DIMENSIONALLY-WEIRD just throw exception in 3D */
template <>
void NematicSystemMPI<2>::
output_defect_positions(const MPI_Comm &mpi_communicator,
                        const std::string data_folder,
                        const std::string filename)
{
    std::vector<std::string> datanames = {"t", "x", "y"};
    datanames.push_back("charge");

    Output::distributed_vector_to_hdf5(defect_pts, 
                                       datanames, 
                                       mpi_communicator, 
                                       data_folder + filename 
                                       + std::string(".h5"));
}



template <>
void NematicSystemMPI<3>::
output_defect_positions(const MPI_Comm &mpi_communicator,
                        const std::string data_folder,
                        const std::string filename)
{
    throw std::logic_error("output_defect_positions not implemented in 3D");
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
               const int time_step)
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

    SingularPotentialPostprocessor<dim>
        singular_potential_postprocessor(lagrange_multiplier);

    DebuggingL3TermPostprocessor<dim> debugging_L3_term_postprocessor;

    DisclinationChargePostprocessor<dim> disclination_charge_postprocessor;

    dealii::DataOut<dim> data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, nematic_postprocessor);
    data_out.add_data_vector(current_solution, energy_postprocessor);
    data_out.add_data_vector(current_solution, configuration_force_postprocessor);
    data_out.add_data_vector(current_solution, singular_potential_postprocessor);
    data_out.add_data_vector(current_solution, debugging_L3_term_postprocessor);
    // data_out.add_data_vector(current_solution, disclination_charge_postprocessor);
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
const LA::MPI::Vector &
NematicSystemMPI<dim>::return_residual() const
{
    return system_rhs;
}



template <int dim>
const LA::MPI::Vector &
NematicSystemMPI<dim>::return_past_solution() const
{
    return past_solution;
}



template <int dim>
const dealii::AffineConstraints<double>&
NematicSystemMPI<dim>::return_constraints() const
{
    return constraints;
}



template <int dim>
double NematicSystemMPI<dim>::return_parameters() const
{
    return maier_saupe_alpha;
}



/** DIMENSIONALLY-WEIRD depends on projection into x-y plane */
template <int dim>
const std::vector<dealii::Point<dim>>& NematicSystemMPI<dim>::
return_initial_defect_pts() const
{
    return initial_value_func->return_defect_pts();
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



template <int dim>
void NematicSystemMPI<dim>::
set_past_solution(const MPI_Comm &mpi_communicator,
                     const LA::MPI::Vector &distributed_solution)
{
    past_solution.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);
    past_solution = distributed_solution;
}



template <int dim>
const std::vector<std::vector<double>>& NematicSystemMPI<dim>::
get_energy_vals()
{
    return energy_vals;
}



template <int dim>
const std::vector<std::vector<double>>& NematicSystemMPI<dim>::
get_defect_pts()
{
    return defect_pts;
}



template <int dim>
void NematicSystemMPI<dim>::
set_energy_vals(const std::vector<std::vector<double>> &energy)
{
    energy_vals = energy;
}



template <int dim>
void NematicSystemMPI<dim>::
set_defect_pts(const std::vector<std::vector<double>> &defects)
{
    defect_pts = defects;
}



/* DIMENSIONALL-DEPENDENT */
template <int dim>
void NematicSystemMPI<dim>::
perturb_configuration_with_director(const MPI_Comm& mpi_communicator,
                                    const dealii::DoFHandler<dim> &director_dof_handler,
                                    const LA::MPI::Vector &director_perturbation)
{
    const dealii::FESystem<dim> director_fe = director_dof_handler.get_fe();
    dealii::Quadrature<dim> quadrature_formula(fe.get_unit_support_points());

    dealii::FEValues<dim> director_fe_values(director_fe,
                                             quadrature_formula,
                                             dealii::update_values);
    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<double> director_vals(n_q_points);
    std::vector<dealii::Vector<double>> 
        Q_vals(n_q_points, dealii::Vector<double>(fe.n_components()));
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> local_solution(dofs_per_cell);

    auto cell = dof_handler.begin_active();
    const auto endc = dof_handler.end();
    auto director_cell = director_dof_handler.begin_active();

    const double eps = 0.1;
    const double mysterious_scale_factor = 1.0;

    LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                           mpi_communicator);
    for (; cell != endc; ++cell, ++director_cell)
    {
        if ( !cell->is_locally_owned() )
            continue;

        director_fe_values.reinit(director_cell);
        fe_values.reinit(cell);
        director_fe_values.get_function_values(director_perturbation,
                                               director_vals);
        fe_values.get_function_values(current_solution, Q_vals);

        // rotate Q_vals by perturbation at every quadrature point
        dealii::Tensor<2, 3> R;
        dealii::SymmetricTensor<2, 3> Q;
        dealii::Tensor<2, 3> Q_rot;
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            double theta = eps * mysterious_scale_factor * director_vals[q];
            R[0][0] = std::cos(theta);
            R[0][1] = -std::sin(theta);
            R[1][0] = std::sin(theta);
            R[1][1] = std::cos(theta);
            R[2][2] = 1.0;

            Q[0][0] = Q_vals[q][0];
            Q[0][1] = Q_vals[q][1];
            Q[0][2] = Q_vals[q][2];
            Q[1][1] = Q_vals[q][3];
            Q[1][2] = Q_vals[q][4];
            Q[2][2] = -(Q[0][0] + Q[1][1]);

            Q_rot = R * Q * dealii::transpose(R);
            Q_vals[q][0] = Q_rot[0][0];
            Q_vals[q][1] = Q_rot[0][1];
            Q_vals[q][2] = Q_rot[0][2];
            Q_vals[q][3] = Q_rot[1][1];
            Q_vals[q][4] = Q_rot[1][2];
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const unsigned int component_i = fe.system_to_component_index(i).first;

            if (component_i == 0)
                local_solution[i] = Q_vals[i][0];
            else if (component_i == 1)
                local_solution[i] = Q_vals[i][1];
            else if (component_i == 2)
                local_solution[i] = Q_vals[i][2];
            else if (component_i == 3)
                local_solution[i] = Q_vals[i][3];
            else if (component_i == 4)
                local_solution[i] = Q_vals[i][4];
            else
                throw std::runtime_error("in perturb_configuration_with_director"
                                         " have gotten invalid component");                   
        }

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            locally_owned_solution[local_dof_indices[i]] = local_solution[i];
    }

    dealii::AffineConstraints<double> configuration_constraints;
    configuration_constraints.clear();
    configuration_constraints.reinit(locally_relevant_dofs);
    dealii::DoFTools::
        make_hanging_node_constraints(dof_handler,
                                      configuration_constraints);

    for (auto const& [boundary_id, boundary_func] : boundary_value_funcs)
        if (boundary_func->return_boundary_condition() == std::string("Dirichlet"))
            dealii::VectorTools::
                interpolate_boundary_values(dof_handler,
                                            boundary_id,
                                            *boundary_func,
                                            configuration_constraints);

    /* WARNING: DEPENDS ON PREVIOUSLY SETTING TRIANGULATION */
    // freeze defects if it's marked on the triangulation
    std::map<dealii::types::material_id, const dealii::Function<dim>*>
        function_map;

    function_map[1] = left_internal_boundary_func.get();
    function_map[2] = right_internal_boundary_func.get();

    SetDefectBoundaryConstraints::
        interpolate_boundary_values(mpi_communicator,
                                    dof_handler, 
                                    function_map, 
                                    configuration_constraints);
    configuration_constraints.close();
    configuration_constraints.distribute(locally_owned_solution);

    locally_owned_solution.compress(dealii::VectorOperation::insert);
    current_solution = locally_owned_solution;
    past_solution = locally_owned_solution;
}

template class NematicSystemMPI<2>;
template class NematicSystemMPI<3>;
