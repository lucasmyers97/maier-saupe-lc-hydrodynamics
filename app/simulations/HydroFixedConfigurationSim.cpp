#include "ExampleFunctions/PlusHalfQTensor.hpp"
#include "SimulationDrivers/BasicHydroDriver.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <memory>

#include "ExampleFunctions/PlusHalfQTensor.hpp"

int main(int ac, char* av[])
{
    try
    {
        const int dim = 2;

        std::unique_ptr<dealii::TensorFunction<2, dim, double>>
            stress_tensor = std::make_unique<PlusHalfQTensor<dim>>();
        BasicHydroDriver<dim> hydro_driver(std::move(stress_tensor));
        hydro_driver.run();
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
