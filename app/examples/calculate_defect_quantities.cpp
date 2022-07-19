#include "Numerics/FindLocalMinima.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>

#include <highfive/H5Easy.hpp>

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

        auto n_cells = dof_handler.get_triangulation().n_active_cells();
        std::vector<double> x(n_cells);
        std::vector<double> y(n_cells);
        std::vector<double> S(n_cells);
        for (dealii::types::global_dof_index i = 0; i < n_cells; ++i)
        {
            x[i] = defect_quantities[i].min_pt[0];
            y[i] = defect_quantities[i].min_pt[1];
            S[i] = defect_quantities[i].min_S;
        }

        std::string filename("defect_quantities.h5");
        H5Easy::File file(filename, H5Easy::File::Create);
        H5Easy::dump(file, "/x", x);
        H5Easy::dump(file, "/y", y);
        H5Easy::dump(file, "/S", S);

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
