#include "Serialization.hpp"

#include <deal.II/base/mpi.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>
#include <fstream>
#include <string>

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"

namespace Serialization
{

    template <int dim>
    void serialize_nematic_system(const MPI_Comm &mpi_communicator,
                                  const std::string filename,
                                  const unsigned int degree,
                                  const dealii::Triangulation<dim> &coarse_tria,
                                  const dealii::parallel::distributed::
                                  Triangulation<dim> &tria,
                                  const NematicSystemMPI<dim> &nematic_system)
    {
        const unsigned int my_id =
            dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
        if (my_id == 0)
        {
            std::ofstream ofs(filename + std::string(".params.ar"));
            boost::archive::text_oarchive oa(ofs);

            oa << degree;
            oa << coarse_tria;
            oa << nematic_system;
        }

        dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
            sol_trans(nematic_system.return_dof_handler());
        sol_trans.prepare_for_serialization(nematic_system.return_current_solution());
        tria.save(filename + std::string(".mesh.ar"));
    }


    template <int dim>
    std::unique_ptr<NematicSystemMPI<dim>>
    deserialize_nematic_system(const MPI_Comm &mpi_communicator,
                               const std::string filename,
                               unsigned int &degree,
                               dealii::Triangulation<dim> &coarse_tria,
                               dealii::parallel::distributed::Triangulation<dim>
                               &tria,
                               const std::string time_discretization)
    {
        std::ifstream ifs(filename + std::string(".params.ar"));
        boost::archive::text_iarchive ia(ifs);

        ia >> degree;
        ia >> coarse_tria;
        tria.copy_triangulation(coarse_tria);

        std::unique_ptr<NematicSystemMPI<dim>> nematic_system
            = std::make_unique<NematicSystemMPI<dim>>(tria, degree);
        ia >> (*nematic_system);

        tria.load(filename + std::string(".mesh.ar"));
        nematic_system->setup_dofs(mpi_communicator, 
                                   /*initial_step=*/true,
                                   time_discretization);

        const dealii::DoFHandler<dim>& dof_handler
            = nematic_system->return_dof_handler();
        const dealii::IndexSet locally_owned_dofs
            = dof_handler.locally_owned_dofs();
        LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                        mpi_communicator);
        dealii::parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
            sol_trans(dof_handler);
        sol_trans.deserialize(completely_distributed_solution);

        nematic_system->set_current_solution(mpi_communicator,
                                             completely_distributed_solution);
        nematic_system->set_past_solution_to_current(mpi_communicator);

        return std::move(nematic_system);
    }

    template
    void serialize_nematic_system<2>(const MPI_Comm &mpi_communicator,
                                  const std::string filename,
                                  const unsigned int degree,
                                  const dealii::Triangulation<2> &coarse_tria,
                                  const dealii::parallel::distributed::
                                  Triangulation<2> &tria,
                                  const NematicSystemMPI<2>
                                  &nematic_system);
    template
    void serialize_nematic_system<3>(const MPI_Comm &mpi_communicator,
                                  const std::string filename,
                                  const unsigned int degree,
                                  const dealii::Triangulation<3> &coarse_tria,
                                  const dealii::parallel::distributed::
                                  Triangulation<3> &tria,
                                  const NematicSystemMPI<3>
                                  &nematic_system);
    template
    std::unique_ptr<NematicSystemMPI<2>>
    deserialize_nematic_system<2>(const MPI_Comm &mpi_communicator,
                                  const std::string filename,
                                  unsigned int &degree,
                                  dealii::Triangulation<2> &coarse_tria,
                                  dealii::parallel::distributed::
                                  Triangulation<2> &tria,
                                  const std::string time_discretization);
    template
    std::unique_ptr<NematicSystemMPI<3>>
    deserialize_nematic_system<3>(const MPI_Comm &mpi_communicator,
                                  const std::string filename,
                                  unsigned int &degree,
                                  dealii::Triangulation<3> &coarse_tria,
                                  dealii::parallel::distributed::
                                  Triangulation<3> &tria,
                                  const std::string time_discretization);
} // namespace Serialization
