#include "ExampleFunctions/PlusHalfQTensor.hpp"
#include "ExampleFunctions/TwoDefectQTensor.hpp"
#include "SimulationDrivers/HydroSystemMPIDriver.hpp"

#include <deal.II/base/utilities.h>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>

#include <memory>
#include <vector>

#include "ExampleFunctions/PlusHalfQTensor.hpp"
#include "ExampleFunctions/PeriodicMu2StressTensor.hpp"
#include "ExampleFunctions/PeriodicQTensor.hpp"
#include "ExampleFunctions/TwoDefectQTensor.hpp"

int main(int ac, char* av[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

        const int dim = 2;
        const unsigned int num_refines = 8;
        const double left = -0.5;
        const double right = 1.5;

        const double A = -0.064;
        const double B = -1.57;
        const double C = 1.29;
        const double k = 5.0;
        const double eps = 0.01;

        const double r = 1.0;
        std::vector<dealii::Point<dim>> p(2);
        p[0][0] = 1.0;
        p[0][1] = 0.0;
        p[1][0] = -1.0;
        p[1][1] = 0.0;

        // std::unique_ptr<dealii::TensorFunction<2, dim, double>>
        //     stress_tensor = std::make_unique<PlusHalfQTensor<dim>>();
        std::unique_ptr<dealii::TensorFunction<2, dim, double>> stress_tensor
            = std::make_unique<PeriodicMu2StressTensor<dim>>(A, B, C, k, eps);
        // std::unique_ptr<dealii::TensorFunction<2, dim, double>> Q_tensor
        //     = std::make_unique<PeriodicQTensor<dim>>(k, eps);
        // std::unique_ptr<dealii::TensorFunction<2, dim, double>> Q_tensor =
        //     std::make_unique<TwoDefectQTensor<dim>>(p, r);
        std::unique_ptr<dealii::TensorFunction<2, dim, double>> Q_tensor =
            std::make_unique<dealii::ZeroTensorFunction<2, dim>>();



        HydroSystemMPIDriver<dim> hydro_driver(std::move(stress_tensor),
                                               std::move(Q_tensor),
                                               num_refines,
                                               left,
                                               right);
        // hydro_driver.run();
        hydro_driver.run_coupled();
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
