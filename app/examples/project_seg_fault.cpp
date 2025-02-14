#include <deal.II/base/utilities.h>

#include <deal.II/base/mpi.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA = dealii::LinearAlgebraTrilinos;

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <vector>
#include <string>

#define VEC_DIM 5

template <int dim>
class UniformFunction : public dealii::Function<dim>
{
public:
    UniformFunction()
        : dealii::Function<dim>(VEC_DIM)
    {}

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        double output = static_cast<double>(component);
        return component;
    }
};

int main(int ac, char *av[])
{
    const int dim = 2;
    const double left = -1;
    const double right = 1;
    const int num_refines = 6;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

    // setup domain
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::FESystem<dim> fe(dealii::FE_Q<dim>(1), VEC_DIM);
    dealii::DoFHandler<dim> dof_handler(triangulation);

    // generate grid
    dealii::GridGenerator::hyper_cube(triangulation, left, right);
    triangulation.refine_global(num_refines);
    dof_handler.distribute_dofs(fe);

    // set up constraints
    dealii::IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    dealii::AffineConstraints<double> locally_owned_constraints;
    locally_owned_constraints.clear();
    locally_owned_constraints.reinit(locally_owned_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                    locally_owned_constraints);

    // project onto vector
    LA::MPI::Vector locally_owned_solution(locally_owned_dofs,
                                           mpi_communicator);
    // dealii::VectorTools::project(dof_handler,
    //                              locally_owned_constraints,
    //                              dealii::QGauss<dim>(fe.degree + 1),
    //                              UniformFunction<dim>(),
    //                              locally_owned_solution);
    dealii::VectorTools::interpolate(dof_handler,
                                     UniformFunction<dim>(),
                                     locally_owned_solution);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names(VEC_DIM);
    for (int i = 0; i < VEC_DIM; ++i)
        solution_names[i] = "component-" + std::to_string(i);
    data_out.add_data_vector(locally_owned_solution, solution_names);
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record("./", "project_debug", 1, mpi_communicator, 2);

    return 0;
}
