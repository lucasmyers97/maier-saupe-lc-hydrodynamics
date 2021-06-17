#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include "sphere_lebedev_rule.hpp"

// Have to put these here -- quirk of C++11
constexpr int LagrangeMultiplier::i[];
constexpr int LagrangeMultiplier::j[];
constexpr int LagrangeMultiplier::mat_dim;
constexpr int LagrangeMultiplier::order;

const dealii::FullMatrix<double>
LagrangeMultiplier::lebedev_coords = makeLebedevCoords();

const dealii::Vector<double>
LagrangeMultiplier::lebedev_weights = makeLebedevWeights();

LagrangeMultiplier::LagrangeMultiplier(double in_alpha=1)
: alpha(in_alpha)
{
    assert(alpha <= 1);
}

dealii::FullMatrix<double>
LagrangeMultiplier::makeLebedevCoords()
{
    double *w = new double[order];
    double *coords = new double[3*order];

    ld_by_order(order, &coords[0], &coords[order], &coords[2*order], w);

    dealii::FullMatrix<double> coord_mat(order, mat_dim, coords); 

    delete w;
    delete coords;
    return coord_mat;
}

dealii::Vector<double>
LagrangeMultiplier::makeLebedevWeights()
{
    double *w = new double[order];
    double *coords = new double[3*order];

    ld_by_order(order, &coords[0], &coords[order], &coords[2*order], w);

    dealii::Vector<double> weight_mat(w, w + order);

    delete w;
    delete coords;
    return weight_mat;
}


double LagrangeMultiplier::sphereIntegral(
        std::function<double (double, double, double)> integrand)
{
    double *x, *y, *z, *w;

	x = new double[order];
	y = new double[order];
	z = new double[order];
	w = new double[order];
    
    ld_by_order(order, x, y, z, w);
    
    double integral;
    for (int k=0; k<order; ++k) {
        integral += w[k]*integrand(x[k], y[k], z[k]);
    }
    integral *= 4*M_PI;

    return integral;
}

void LagrangeMultiplier::printVecTest()
{
    double *w = new double[order];
    double *coords = new double[3*order];

    ld_by_order(order, &coords[0], &coords[order], &coords[2*order], w);

//    dealii::FullMatrix<double>::Accessor a(&lebedev_coords, 0, 0);
    auto iter = lebedev_coords.begin();
    ++iter;
    auto access = *iter;
    auto num = access.value();
    std::cout << num << std::endl;
    std::cout << coords[1] << std::endl;

//    std::cout << "Lebedev coordinate is: " << a.value() << std::endl;
//    std::cout << "Lebedev coordinates are: ";
//    lebedev_coords.print(std::cout);
//    std::cout << std::endl;

//    std::cout << "Lebedev weights are: " <<
//        lebedev_weights << std::endl;

    delete w;
    delete coords;
}

