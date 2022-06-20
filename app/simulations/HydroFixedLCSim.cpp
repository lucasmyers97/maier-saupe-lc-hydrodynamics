#include "SimulationDrivers/HydroFixedLCDriver.hpp"

int main()
{
    const int dim = 2;
    HydroFixedLCDriver<dim> hydro_fixed_driver;

    hydro_fixed_driver.run();

    return 0;
}
