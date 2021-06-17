#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <eigen3/Eigen/Dense>
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
    return coord_mat;
}

dealii::Vector<double>
LagrangeMultiplier::makeLebedevWeights()
{
    double *w = new double[order];
    double *coords = new double[3*order];

    ld_by_order(order, &coords[0], &coords[order], &coords[2*order], w);


    dealii::Vector<double> weight_mat(w, w + order);
    return weight_mat;
}


double LagrangeMultiplier::sphereIntegral(
        std::function<double (double, double, double)> integrand)
{
    double *x;
	double *y;
	double *z;
	double *w;

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



// void LagrangeMultiplier::Test(
//         std::function<double (double, double, double)> f)
// {
//     Matrix<int, mat_dim, mat_dim> m;
//     for (int k=0; k<vec_dim; ++k) {
//         m(i[k], j[k]) = k;
//         if (i[k] != j[k]) {
//             m(j[k], i[k]) = k;
//         }
//     }
// 
//     std::cout << m << std::endl;
//     std::cout << alpha << std::endl;
// 
//     double integral = sphereIntegral(f);
// 
//     std::cout << "Integral is:\n" << integral << std::endl;
// 
//     std::cout << lebedev_coords << std::endl;
//     std::cout << lebedev_weights << std::endl;
// 
// }

void LagrangeMultiplier::printVecTest()
{
    lebedev_coords.print(std::cout);
    std::cout << lebedev_weights << std::endl;
}

