#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Utilities/Serialization.hpp"
#include "Postprocessors/DistortionStressPostprocessor.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>

#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

int main(int ac, char* av[])
{
    try
    {
        if (ac != 2)
            throw std::invalid_argument("Error! Need to input filename!");

        std::string filename(av[1]);

        const int dim = 2;

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);
        MPI_Comm mpi_communicator(MPI_COMM_WORLD);

        unsigned int degree;
        dealii::Triangulation<dim> coarse_tria;
        dealii::parallel::distributed::Triangulation<dim> tria(mpi_communicator);

        std::unique_ptr<NematicSystemMPI<dim>> nematic_system
            = Serialization::deserialize_nematic_system(mpi_communicator,
                                                        filename,
                                                        degree,
                                                        coarse_tria,
                                                        tria);

        const dealii::TrilinosWrappers::MPI::Vector &solution
            = nematic_system->return_current_solution();
        const dealii::DoFHandler<dim> &dof_handler
            = nematic_system->return_dof_handler();

        DistortionStressPostprocessor<dim> distortion_postprocessor;
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        std::vector<std::string> Q_names(msc::vec_dim<dim>);
        for (std::size_t i = 0; i < Q_names.size(); ++i)
            Q_names[i] = std::string("Q") + std::to_string(i);

        data_out.add_data_vector(solution, Q_names);
        data_out.add_data_vector(solution, distortion_postprocessor);
        data_out.build_patches();

        std::string output_filename("periodic_Q_components.vtu");
        std::ofstream output(output_filename);
        data_out.write_vtu(output);

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        return 1;
    }
}
