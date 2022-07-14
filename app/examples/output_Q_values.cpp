#include "LiquidCrystalSystems/NematicSystemMPI.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Postprocessors/DisclinationChargePostprocessor.hpp"
#include "Postprocessors/DistortionStressPostprocessor.hpp"
#include "Postprocessors/MolecularFieldPostprocessor.hpp"
#include "Postprocessors/mu1StressPostprocessor.hpp"
#include "Utilities/Serialization.hpp"

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
        if (ac != 3)
            throw std::invalid_argument("Error! Need to enter input and" 
                                        "output filename filename!");

        std::string input_filename(av[1]);
        std::string output_filename(av[2]);

        const int dim = 2;
        const int order = 974;
        const double lagrange_alpha = 1.0;
        const double tol = 1e-10;
        const unsigned int max_iter = 20;
        LagrangeMultiplierAnalytic<dim>
            lma(order, lagrange_alpha, tol, max_iter);
        const double alpha = 8.0;

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

        DisclinationChargePostprocessor<dim> disclination_postprocessor;
        DistortionStressPostprocessor<dim> distortion_postprocessor;
        mu1StressPostprocessor<dim> mu1_postprocessor(lma, alpha);
        MolecularFieldPostprocessor<dim> molecular_postprocessor(lma, alpha);
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        std::vector<std::string> Q_names(msc::vec_dim<dim>);
        for (std::size_t i = 0; i < Q_names.size(); ++i)
            Q_names[i] = std::string("Q") + std::to_string(i);

        data_out.add_data_vector(solution, Q_names);
        data_out.add_data_vector(solution, disclination_postprocessor);
        data_out.add_data_vector(solution, distortion_postprocessor);
        data_out.add_data_vector(solution, mu1_postprocessor);
        data_out.add_data_vector(solution, molecular_postprocessor);
        data_out.build_patches();

        std::ofstream output(output_filename);
        data_out.write_vtu(output);

        return 0;
    }
    catch (std::exception &exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
