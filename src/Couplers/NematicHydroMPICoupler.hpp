#ifndef NEMATIC_HYDRO_MPI_COUPLER_HPP
#define NEMATIC_HYDRO_MPI_COUPLER_HPP

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "LiquidCrystalSystems/HydroSystemMPI.hpp"

template <int dim>
class NematicHydroMPICoupler
{
public:
    NematicHydroMPICoupler(){};

    void assemble_hydro_system(const NematicSystemMPI<dim> &nematic_system,
                               HydroSystemMPI<dim> &hydro_system);
    // void assemble_nematic_hydro_system(NematicSystemMPI<dim> &nematic_system,
    //                                    HydroSystemMPI<dim> &hydro_system);
};

#endif
