#include "BoundaryValues/DzyaloshinskiiFunction.hpp"

#include <deal.II/base/point.h>

int main()
{
    const int dim = 2;

    dealii::Point<dim> p({2.0, 0.0});
    double S0 = 0.6751;
    double eps = 0.1;
    unsigned int degree = 1;
    double charge = 0.5;
    unsigned int n_refines = 10;
    double tol = 1e-10;
    unsigned int max_iter = 100;
    double newton_step = 1.0;



    DzyaloshinskiiFunction<dim> 
        df(p, S0, eps, degree, charge, n_refines, tol, max_iter, newton_step);
    
    df.initialize();

    return 0;
}
