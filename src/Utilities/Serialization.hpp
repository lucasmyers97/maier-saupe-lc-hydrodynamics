#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <deal.II/base/mpi.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>

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
                                  const NematicSystemMPI<dim> &nematic_system);

    template <int dim>
    std::unique_ptr<NematicSystemMPI<dim>>
    deserialize_nematic_system(const MPI_Comm &mpi_communicator,
                               const std::string filename,
                               unsigned int &degree,
                               dealii::Triangulation<dim> &coarse_tria,
                               dealii::parallel::distributed::
                               Triangulation<dim> &tria,
                               const std::string time_discretization
                               = std::string("convex_splitting"));
} // end namespace serialization

#endif
