#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA = dealii::LinearAlgebraPETSc;

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/vector_operation.h>

#include <iostream>
#include <vector>

const int dim = 2;

const double left = -1.0;
const double right = 1.0;
const int num_refines = 2;

int main(int ac, char *av[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

    // Set up mpi objects
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    int num_ranks = 0;
    MPI_Comm_size(mpi_communicator, &num_ranks);
    dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

    // To differentiate between running with one vs. more ranks
    pcout << "num_ranks: " << std::to_string(num_ranks) << "\n\n";

    // Declare other necessary finite element objects
    dealii::parallel::distributed::Triangulation<dim>
        triangulation(mpi_communicator,
                      typename dealii::Triangulation<dim>::MeshSmoothing(
                          dealii::Triangulation<dim>::smoothing_on_refinement |
                          dealii::Triangulation<dim>::smoothing_on_coarsening));
    dealii::DoFHandler<dim> dof_handler(triangulation);
    dealii::FE_Q<dim> fe(1);

    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;
    LA::MPI::Vector locally_relevant_solution;
    LA::MPI::Vector locally_owned_solution;

    // Generate grid
    dealii::GridGenerator::hyper_cube(triangulation, left, right);
    triangulation.refine_global(num_refines);

    // Initialize dofs and vectors
    dof_handler.distribute_dofs(fe);

    // Set up all relevant vectors
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);
    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    locally_owned_solution.reinit(locally_owned_dofs,
                                  mpi_communicator);

    // Set all values to 1
    dealii::VectorTools::interpolate(dof_handler,
                                     dealii::ConstantFunction<dim>(1),
                                     locally_owned_solution);
    // Compress here because interpolating inserts elements into the vector
    locally_owned_solution.compress(dealii::VectorOperation::insert);

    // Fine because we're setting ghost vector with completely distributed vector
    locally_relevant_solution = locally_owned_solution;
    locally_relevant_solution.compress(dealii::VectorOperation::insert);

    pcout << "Printing ghost elements directly "
          << "set by completely distributed vector" << "\n";
    for (auto dof : locally_relevant_dofs)
      pcout << locally_relevant_solution[dof] << "\n";
    pcout << "\n";

    // Not okay because we are trying to modify a ghost vector
    // Should throw warning here
    locally_relevant_solution += locally_owned_solution;
    // Still bad even if we compress, but we do so anyways for good measure
    locally_relevant_solution.compress(dealii::VectorOperation::add);

    pcout << "Printing incorrectly modified ghost elements" << "\n";
    for (auto dof : locally_relevant_dofs)
      pcout << locally_relevant_solution[dof] << "\n";

    return 0;
}
