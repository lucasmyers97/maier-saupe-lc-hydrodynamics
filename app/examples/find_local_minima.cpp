/**
 * This program reads in a serialization of a NematicSystemMPI from the
 * first command-line argument, and then goes through and calculates the
 * defect quantities (these are min_S, max_D, and point of min_S for every
 * cell).
 *
 * Then, just to check that everything is right, it outputs the min S and
 * corresponding points to an hdf5 file so that one can plot and make sure it
 * looks like the given vtu file
 */

#include "Numerics/FindLocalMinima.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_out.h>

#include <highfive/H5Easy.hpp>

#include <string>

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Utilities/Serialization.hpp"

#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Utilities/Serialization.hpp"

int main(int ac, char* av[])
{
    try
    {
        if (ac != 2)
            throw std::invalid_argument("Error! Need to enter input filename");

        std::string input_filename(av[1]);

        const int dim = 2;

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        unsigned int degree;
        dealii::Triangulation<dim> coarse_tria;
        dealii::parallel::distributed::Triangulation<dim> tria(mpi_communicator);

        std::unique_ptr<NematicSystemMPI<dim>> nematic_system
            = Serialization::deserialize_nematic_system(mpi_communicator,
                                                        input_filename,
                                                        degree,
                                                        coarse_tria,
                                                        tria);

        const dealii::TrilinosWrappers::MPI::Vector &solution
            = nematic_system->return_current_solution();
        const dealii::DoFHandler<dim> &dof_handler
            = nematic_system->return_dof_handler();

        std::vector<NumericalTools::DefectQuantities<dim>> defect_quantities
            = NumericalTools::calculate_defect_quantities<dim>(dof_handler, 
                                                               solution);

        double R = 2.0;
        bool is_local_min = false;
        unsigned int idx = 0;

        for (auto cell = dof_handler.begin_active(); 
             cell != dof_handler.end(); ++cell)
        {
            // if (cell->user_flag_set())
            //     cell->set_material_id(1);
            // else
            //     cell->set_material_id(2);
            is_local_min 
                = NumericalTools::check_if_local_min<dim>(cell, 
                                                          R, 
                                                          defect_quantities);

            idx = cell->user_index();
            if (is_local_min)
                std::cout << "Point is: " << defect_quantities[idx].min_pt
                          << "\n\n";
        }

        std::ofstream out("find_minima_grid.svg");
        dealii::GridOut grid_out;
        dealii::GridOutFlags::Svg flags;
        flags.coloring = dealii::GridOutFlags::Svg::Coloring::material_id;
        grid_out.set_flags(flags);
        grid_out.write_svg(tria, out);

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
