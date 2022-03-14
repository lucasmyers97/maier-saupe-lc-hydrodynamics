#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparse_ilu.h>

#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/numerics/vector_tools_boundary.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>

#include "Utilities/maier_saupe_constants.hpp"
#include "Utilities/SimulationOptions.hpp"
#include "LiquidCrystalSystems/IsoTimeDependent.hpp"
#include "BoundaryValues/BoundaryValues.hpp"
#include "BoundaryValues/BoundaryValuesFactory.hpp"

namespace msc = maier_saupe_constants;
namespace po = boost::program_options;

const int order = 974;

template <int dim>
struct InnerPreconditioner;

template <>
struct InnerPreconditioner<2>
{
    using type = dealii::SparseDirectUMFPACK;
};

template <>
struct InnerPreconditioner<3>
{
    using type = dealii::SparseILU<double>;
};

template <int dim>
class HydroFixedConfiguration
{
public:
    HydroFixedConfiguration(const unsigned int degree,
                            const po::variables_map vm);
    void run(IsoTimeDependent<dim, order> &iso_time_dependent);

private:
    void setup_dofs();
    void assemble_system();
    void solve();
    void output_results(const unsigned int refinement_cycle) const;
    void refine_mesh();
    void project_Q_tensor(dealii::Functions::FEFieldFunction<dim> &fe_field);

    const unsigned int degree;

    dealii::Triangulation<dim> triangulation;
    dealii::FESystem<dim>      fe;
    dealii::DoFHandler<dim>    dof_handler;

    dealii::AffineConstraints<double> constraints;
    std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

    dealii::BlockSparsityPattern      sparsity_pattern;
    dealii::BlockSparseMatrix<double> system_matrix;

    dealii::BlockSparsityPattern      preconditioner_sparsity_pattern;
    dealii::BlockSparseMatrix<double> preconditioner_matrix;

    dealii::BlockVector<double> solution;
    dealii::BlockVector<double> system_rhs;

    std::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
};



template <int dim>
class QTensorProject : public dealii::Function<dim>
{
public:
    QTensorProject(dealii::Functions::FEFieldFunction<dim> &ext_field)
        : dealii::Function<dim>(dim + 1 + msc::vec_dim<dim>),
        fe_field(ext_field)
    {}

    dealii::Functions::FEFieldFunction<dim> &fe_field;
    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        if (component < dim + 1) {
            return 0;
        } else {
            return fe_field.value(p, component - (dim + 1));
        }
    }

    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &value) const override
    {
        for (int i = 0; i < value.size(); ++i)
        {
            if (i < dim + 1) {
                value[i] = 0;
            } else {
                value[i] = fe_field.value(p, i - (dim + 1));
            }
        }
    }

};



template <int dim>
class HydroBoundaryValues : public dealii::Function<dim>
{
public:
    HydroBoundaryValues()
        : dealii::Function<dim>(dim + 1 + msc::vec_dim<dim>)
    {}

    virtual double value(const dealii::Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &  value) const override;
};


template <int dim>
double HydroBoundaryValues<dim>::value(const dealii::Point<dim> & p,
                                  const unsigned int component) const
{
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    // if (component == 0)
    //   return (p[0] < 0 ? -1 : (p[0] > 0 ? 1 : 0));
    // if (component == 0 || component == 1)
    return 0;
}


template <int dim>
void HydroBoundaryValues<dim>::vector_value(const dealii::Point<dim> &p,
                                       dealii::Vector<double> &  values) const
{
    for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = HydroBoundaryValues<dim>::value(p, c);
}

template <int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
    RightHandSide()
        : dealii::Function<dim>(dim + 1 + msc::vec_dim<dim>)
    {}

    virtual double value(const dealii::Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &  value) const override;
};


template <int dim>
double RightHandSide<dim>::value(const dealii::Point<dim> &p /*p*/,
                                 const unsigned int component /*component*/) const
{
    if (component == 0)
        return 1 / (2 * std::sqrt(p[0]*p[0] + p[1]*p[1]));
    else
        return 0;
}


template <int dim>
void RightHandSide<dim>::vector_value(const dealii::Point<dim> &p,
                                      dealii::Vector<double> &  values) const
{
    for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = RightHandSide<dim>::value(p, c);
}


template <class MatrixType, class PreconditionerType>
class InverseMatrix : public dealii::Subscriptor
{
public:
    InverseMatrix(const MatrixType &        m,
                  const PreconditionerType &preconditioner);

    void vmult(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src) const;

private:
    const dealii::SmartPointer<const MatrixType>         matrix;
    const dealii::SmartPointer<const PreconditionerType> preconditioner;
};


template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix
    (const MatrixType &        m,
     const PreconditionerType &preconditioner)
    : matrix(&m)
    , preconditioner(&preconditioner)
{}

template <class MatrixType, class PreconditionerType>
void InverseMatrix<MatrixType, PreconditionerType>::vmult
    (dealii::Vector<double> &      dst,
     const dealii::Vector<double> &src) const
{
    dealii::SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
    dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

    dst = 0;

    cg.solve(*matrix, dst, src, *preconditioner);
}


template <class PreconditionerType>
class SchurComplement : public dealii::Subscriptor
{
public:
    SchurComplement
    (const dealii::BlockSparseMatrix<double> &system_matrix,
     const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType> &A_inverse);

    void vmult(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src) const;

private:
    const dealii::SmartPointer<const dealii::BlockSparseMatrix<double>> system_matrix;
    const dealii::SmartPointer<
        const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType>>
    A_inverse;

    mutable dealii::Vector<double> tmp1, tmp2;
};



template <class PreconditionerType>
SchurComplement<PreconditionerType>::SchurComplement
    (const dealii::BlockSparseMatrix<double> &system_matrix,
     const InverseMatrix<dealii::SparseMatrix<double>, PreconditionerType> &A_inverse)
    : system_matrix(&system_matrix)
    , A_inverse(&A_inverse)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
{}


template <class PreconditionerType>
void SchurComplement<PreconditionerType>::vmult
    (dealii::Vector<double> &      dst,
     const dealii::Vector<double> &src) const
{
    system_matrix->block(0, 1).vmult(tmp1, src);
    A_inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
}

template <int dim>
HydroFixedConfiguration<dim>::HydroFixedConfiguration(const unsigned int degree,
                                                      const po::variables_map vm)
    : degree(degree)
    , triangulation(dealii::Triangulation<dim>::maximum_smoothing)
    , fe(dealii::FE_Q<dim>(degree + 1), dim,
         dealii::FE_Q<dim>(degree), 1,
         dealii::FE_Q<dim>(degree + 1), msc::vec_dim<dim>)
    , dof_handler(triangulation)
    , boundary_value_func(BoundaryValuesFactory::BoundaryValuesFactory<dim>(vm))
{}


template <int dim>
void HydroFixedConfiguration<dim>::setup_dofs()
{
    A_preconditioner.reset();
    system_matrix.clear();
    preconditioner_matrix.clear();

    dof_handler.distribute_dofs(fe);
    dealii::DoFRenumbering::Cuthill_McKee(dof_handler);

    std::vector<unsigned int> block_component(dim + 1 + msc::vec_dim<dim>, 0);
    block_component[dim] = 1;
    for (int i = dim + 1; i < block_component.size(); ++i)
        block_component[i] = 2;
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);

    {
        constraints.clear();

        dealii::FEValuesExtractors::Vector velocities(0);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler,
                                                         0,
                                                         // HydroBoundaryValues<dim>(),
                                                         dealii::Functions::ZeroFunction<dim>(dim + 1 + msc::vec_dim<dim>),
                                                         constraints,
                                                         fe.component_mask(velocities));

        // dealii::FEValuesExtractors::Vector Q_tensor(dim + 1);
        // dealii::VectorTools::interpolate_boundary_values(dof_handler,
        //                                                  0,
        //                                                  *boundary_value_func,
        //                                                  constraints,
        //                                                  fe.component_mask(Q_tensor));
    }

    constraints.close();

    const std::vector<dealii::types::global_dof_index> dofs_per_block =
        dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];
    const unsigned int n_Q = dofs_per_block[2];

    std::cout << "   Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << '+' << n_Q << ')' << std::endl;

    {
        dealii::BlockDynamicSparsityPattern dsp(3, 3);

        dsp.block(0, 0).reinit(n_u, n_u);
        dsp.block(1, 0).reinit(n_p, n_u);
        dsp.block(2, 0).reinit(n_Q, n_u);
        dsp.block(0, 1).reinit(n_u, n_p);
        dsp.block(1, 1).reinit(n_p, n_p);
        dsp.block(2, 1).reinit(n_Q, n_p);
        dsp.block(0, 2).reinit(n_u, n_Q);
        dsp.block(1, 2).reinit(n_p, n_Q);
        dsp.block(2, 2).reinit(n_Q, n_Q);

        dsp.collect_sizes();

        dealii::Table<2, dealii::DoFTools::Coupling>
            coupling(dim + 1 + msc::vec_dim<dim>, dim + 1 + msc::vec_dim<dim>);

        for (unsigned int c = 0; c < dim + 1 + msc::vec_dim<dim>; ++c)
            for (unsigned int d = 0; d < dim + 1 + msc::vec_dim<dim>; ++d)
                if ((c < dim) && (d < dim))
                    coupling[c][d] = dealii::DoFTools::always;
                else if ((c < dim) && (d == dim))
                    coupling[c][d] = dealii::DoFTools::always;
                else if ((c == dim) && (d < dim))
                    coupling[c][d] = dealii::DoFTools::always;
                else
                    coupling[c][d] = dealii::DoFTools::none;


        dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                coupling,
                                                dsp,
                                                constraints,
                                                false);

        sparsity_pattern.copy_from(dsp);
    }

    {
        dealii::BlockDynamicSparsityPattern preconditioner_dsp(3, 3);

        preconditioner_dsp.block(0, 0).reinit(n_u, n_u);
        preconditioner_dsp.block(1, 0).reinit(n_p, n_u);
        preconditioner_dsp.block(2, 0).reinit(n_Q, n_u);
        preconditioner_dsp.block(0, 1).reinit(n_u, n_p);
        preconditioner_dsp.block(1, 1).reinit(n_p, n_p);
        preconditioner_dsp.block(2, 1).reinit(n_Q, n_p);
        preconditioner_dsp.block(0, 2).reinit(n_u, n_Q);
        preconditioner_dsp.block(1, 2).reinit(n_p, n_Q);
        preconditioner_dsp.block(2, 2).reinit(n_Q, n_Q);

        preconditioner_dsp.collect_sizes();

        dealii::Table<2, dealii::DoFTools::Coupling>
            preconditioner_coupling(dim + 1 + msc::vec_dim<dim>,
                                    dim + 1 + msc::vec_dim<dim>);

        for (unsigned int c = 0; c < dim + 1 + msc::vec_dim<dim>; ++c)
          for (unsigned int d = 0; d < dim + 1 + msc::vec_dim<dim>; ++d)
            if ((c == dim) && (d == dim))
              preconditioner_coupling[c][d] = dealii::DoFTools::always;
            else
              preconditioner_coupling[c][d] = dealii::DoFTools::none;

        dealii::DoFTools::make_sparsity_pattern(dof_handler,
                                                preconditioner_coupling,
                                                preconditioner_dsp,
                                                constraints,
                                                false);

        preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    }

    system_matrix.reinit(sparsity_pattern);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

    solution.reinit(3);
    solution.block(0).reinit(n_u);
    solution.block(1).reinit(n_p);
    solution.block(2).reinit(n_Q);
    solution.collect_sizes();

    system_rhs.reinit(3);
    system_rhs.block(0).reinit(n_u);
    system_rhs.block(1).reinit(n_p);
    system_rhs.block(2).reinit(n_Q);
    system_rhs.collect_sizes();
}



template <int dim>
void HydroFixedConfiguration<dim>::
    project_Q_tensor(dealii::Functions::FEFieldFunction<dim> &fe_field)
{
    QTensorProject<dim> q_tensor_project(fe_field);

    std::vector<unsigned int> block_component(dim + 1 + msc::vec_dim<dim>, 0);
    block_component[dim] = 1;
    for (int i = dim + 1; i < block_component.size(); ++i)
      block_component[i] = 2;

    const std::vector<dealii::types::global_dof_index> dofs_per_block =
        dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];
    const unsigned int n_Q = dofs_per_block[2];

    dealii::BlockVector<double> tmp;
    tmp.reinit(3);
    tmp.block(0).reinit(n_u);
    tmp.block(1).reinit(n_p);
    tmp.block(2).reinit(n_Q);
    tmp.collect_sizes();

    dealii::VectorTools::project(dof_handler,
                                 constraints,
                                 dealii::QGauss<dim>(fe.degree + 1),
                                 q_tensor_project,
                                 tmp);

    solution.block(2) = tmp.block(2);
}


template <int dim>
void HydroFixedConfiguration<dim>::assemble_system()
{
    system_matrix         = 0;
    system_rhs            = 0;
    preconditioner_matrix = 0;

    dealii::QGauss<dim> quadrature_formula(degree + 2);

    dealii::FEValues<dim> fe_values(fe,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values |
                                    dealii::update_gradients);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                           dofs_per_cell);
    dealii::Vector<double>     local_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const RightHandSide<dim>    right_hand_side;
    std::vector<dealii::Vector<double>>
        rhs_values(n_q_points,
                   dealii::Vector<double>(dim + 1 + msc::vec_dim<dim>));

    const dealii::FEValuesExtractors::Vector velocities(0);
    const dealii::FEValuesExtractors::Scalar pressure(dim);

    std::vector<dealii::SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    bool printed = false;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix                = 0;
        local_preconditioner_matrix = 0;
        local_rhs                   = 0;

        right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                const unsigned int component_k =
                    fe.system_to_component_index(k).first;
                if (component_k > dim)
                    continue;
                symgrad_phi_u[k] =
                    fe_values[velocities].symmetric_gradient(k, q);
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                phi_p[k]     = fe_values[pressure].value(k, q);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                  fe.system_to_component_index(i).first;
              if (component_i > dim)
                  continue;

              for (unsigned int j = 0; j <= i; ++j)
              {
                  const unsigned int component_j =
                      fe.system_to_component_index(j).first;
                  if (component_j > dim)
                      continue;

                  local_matrix(i, j) +=
                      (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) // (1)
                       - div_phi_u[i] * phi_p[j]                 // (2)
                       - phi_p[i] * div_phi_u[j])                // (3)
                      * fe_values.JxW(q);                        // * dx

                  local_preconditioner_matrix(i, j) +=
                      (phi_p[i] * phi_p[j]) // (4)
                      * fe_values.JxW(q);   // * dx
              }
                local_rhs(i) += (fe_values.shape_value(i, q)   // (phi_u_i(x_q)
                                 * rhs_values[q](component_i)) // * f(x_q))
                    * fe_values.JxW(q);            // * dx
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const unsigned int component_i =
                fe.system_to_component_index(i).first;
            if (component_i > dim)
                continue;
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            {
                const unsigned int component_j =
                    fe.system_to_component_index(j).first;
                if (component_j > dim)
                    continue;
                local_matrix(i, j) = local_matrix(j, i);
                local_preconditioner_matrix(i, j) =
                    local_preconditioner_matrix(j, i);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
        // constraints.distribute_local_to_global(local_matrix,
        //                                        local_dof_indices,
        //                                        preconditioner_matrix);
        constraints.distribute_local_to_global(local_preconditioner_matrix,
                                               local_dof_indices,
                                               preconditioner_matrix);
    }

    std::cout << "   Computing preconditioner..." << std::endl << std::flush;

    A_preconditioner =
        std::make_shared<typename InnerPreconditioner<dim>::type>();
    std::cout << system_matrix.block(0, 0).m() << " by "
              << system_matrix.block(0, 0).n() << std::endl;
    A_preconditioner->factorize(system_matrix.block(0, 0));
    A_preconditioner->initialize(system_matrix.block(0, 0),
                                 typename InnerPreconditioner<dim>::type::AdditionalData());
    std::cout << "done computing preconditioner..." << std::endl;
}


template <int dim>
void HydroFixedConfiguration<dim>::solve()
{
    const InverseMatrix<dealii::SparseMatrix<double>,
                        typename InnerPreconditioner<dim>::type>
        A_inverse(system_matrix.block(0, 0), *A_preconditioner);
    dealii::Vector<double> tmp(solution.block(0).size());

    {
        dealii::Vector<double> schur_rhs(solution.block(1).size());
        A_inverse.vmult(tmp, system_rhs.block(0));
        system_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= system_rhs.block(1);

        SchurComplement<typename InnerPreconditioner<dim>::type>
            schur_complement(system_matrix, A_inverse);

        dealii::SolverControl solver_control(solution.block(1).size(),
                                             1e-6 * schur_rhs.l2_norm());
        dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

        dealii::SparseILU<double> preconditioner;
        preconditioner.initialize(preconditioner_matrix.block(1, 1),
                                  dealii::SparseILU<double>::AdditionalData());

        InverseMatrix<dealii::SparseMatrix<double>, dealii::SparseILU<double>>
            m_inverse(preconditioner_matrix.block(1, 1), preconditioner);

        cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);

        constraints.distribute(solution);

        std::cout << "  " << solver_control.last_step()
                  << " outer CG Schur complement iterations for pressure"
                  << std::endl;
    }

    {
        system_matrix.block(0, 1).vmult(tmp, solution.block(1));
        tmp *= -1;
        tmp += system_rhs.block(0);

        A_inverse.vmult(solution.block(0), tmp);

        constraints.distribute(solution);
    }
}

template <int dim>
void
HydroFixedConfiguration<dim>::output_results(const unsigned int refinement_cycle) const
{
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    solution_names.emplace_back("Q1");
    solution_names.emplace_back("Q2");
    solution_names.emplace_back("Q3");
    solution_names.emplace_back("Q4");
    solution_names.emplace_back("Q5");

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim,
                                      dealii::DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(dealii::DataComponentInterpretation::component_is_scalar);
    for (int i = 0; i < msc::vec_dim<dim>; ++i)
      data_component_interpretation.push_back(
          dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output("solution-"
                         + dealii::Utilities::int_to_string(refinement_cycle, 2)
                         + ".vtu");
    data_out.write_vtu(output);
}

template <int dim>
void HydroFixedConfiguration<dim>::refine_mesh()
{
    dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    dealii::FEValuesExtractors::Scalar pressure(dim);
    dealii::KellyErrorEstimator<dim>::estimate(
                                       dof_handler,
                                       dealii::QGauss<dim - 1>(degree + 1),
                                       std::map<dealii::types::boundary_id, const dealii::Function<dim> *>(),
                                       solution,
                                       estimated_error_per_cell,
                                       fe.component_mask(pressure));

    dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                            estimated_error_per_cell,
                                                            0.3,
                                                            0.0);
    triangulation.execute_coarsening_and_refinement();
}


template <int dim>
void HydroFixedConfiguration<dim>::run
    (IsoTimeDependent<dim, order> &iso_time_dependent)
{
    {
        std::vector<unsigned int> subdivisions(dim, 4);
        // subdivisions[0] = 4;

        // const Point<dim> bottom_left = (dim == 2 ?                //
        //                                   Point<dim>(-2, -1) :    // 2d case
        //                                   Point<dim>(-2, 0, -1)); // 3d case

        // const Point<dim> top_right = (dim == 2 ?              //
        //                                 Point<dim>(2, 0) :    // 2d case
        //                                 Point<dim>(2, 1, 0)); // 3d case
        const dealii::Point<dim> bottom_left = (dim == 2 ?                //
                                                dealii::Point<dim>(-1, -1) :    // 2d case
                                                dealii::Point<dim>(-2, 0, -1)); // 3d case

        const dealii::Point<dim> top_right = (dim == 2 ?              //
                                              dealii::Point<dim>(1, 1) :    // 2d case
                                              dealii::Point<dim>(2, 1, 0)); // 3d case

        // GridGenerator::subdivided_hyper_rectangle(triangulation,
        //                                           subdivisions,
        //                                           bottom_left,
        //                                           top_right);
        dealii::GridGenerator::hyper_cube(triangulation,
                                          -10 / std::sqrt(2),
                                          10 / std::sqrt(2));
    }

    // for (const auto &cell : triangulation.active_cell_iterators())
    //   for (const auto &face : cell->face_iterators())
    //     // if (face->center()[dim - 1] == 0)
    //       face->set_all_boundary_ids(1);


    // triangulation.refine_global(4 - dim);
    triangulation.refine_global(5);
    std::cout << "Done refining mesh" << std::endl;

    for (unsigned int refinement_cycle = 0; refinement_cycle < 1;
         ++refinement_cycle)
        {
            std::cout << "Refinement cycle " << refinement_cycle << std::endl;

            if (refinement_cycle > 0)
                refine_mesh();

            setup_dofs();
            dealii::Functions::FEFieldFunction<dim> fe_field
                = iso_time_dependent.return_fe_field();
            project_Q_tensor(fe_field);

            std::cout << "   Assembling..." << std::endl << std::flush;
            assemble_system();

            std::cout << "   Solving..." << std::flush;
            solve();

            output_results(refinement_cycle);

            std::cout << std::endl;
        }
}


int main(int ac, char* av[])
{
  try
    {
    const int dim = 2;

    po::variables_map vm = SimulationOptions::read_command_line_options(ac, av);
    IsoTimeDependent<dim, order> iso_time_dependent(vm);
    iso_time_dependent.run();

    HydroFixedConfiguration<dim> flow_problem(1, vm);
    flow_problem.run(iso_time_dependent);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
