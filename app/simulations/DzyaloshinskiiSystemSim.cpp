#include "LiquidCrystalSystems/DzyaloshinskiiSystem.hpp"

#include <string>
#include <ostream>

int main(int ac, char* av[])
{
    double eps = -0.9999999;
    unsigned int degree = 1;
    unsigned int n_refines = 8;
    double tol = 1e-8;
    unsigned int max_iter = 100;
    double newton_step = 1.0;

    std::string filename("dzyaloshinskii_solution_");
    filename += std::to_string(eps);
    filename += std::string(".vtu");

    DzyaloshinskiiSystem dzyaloshinskii_system(eps, degree);
    dzyaloshinskii_system.make_grid(n_refines);
    dzyaloshinskii_system.setup_system();
    double res = dzyaloshinskii_system.run_newton_method(tol, max_iter, newton_step);
    dzyaloshinskii_system.output_solution(filename);

    std::cout << res << "\n";

    return 0;
}
