#include "LagrangeMultiplier.hpp"
#include <iostream>
#include <cmath>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/table_indices.h>
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
    // Get weights and coordinates of points around sphere for
    // Lebedev Quadrature
    double *w = new double[order];
    double *coords = new double[3*order];
    ld_by_order(order, &coords[0], &coords[order], &coords[2*order], w);

    // Put in dealii FullMatrix -- need to take transpose because
    // it fills the matrix in a weird way
    dealii::FullMatrix<double> coord_matT(mat_dim, order, coords);
    dealii::FullMatrix<double> coord_mat(order, mat_dim);
    coord_mat.copy_transposed(coord_matT);

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

    // Find difference between lebedev_coords/weights as member variables,
    // and between those ripped directly from ld_by_order
    double sum_coords = 0;
    double sum_weights = 0;
    for (int k = 0; k < mat_dim; ++k) {
        for (int l = 0; l < order; ++l) {
            dealii::TableIndices<2> idx(l, k);
            sum_coords += abs(lebedev_coords(idx) - coords[k*order + l]);
            
            sum_weights += abs(lebedev_weights[k] - w[k]);
        }
    }

    std::cout << "Difference in coords is: " << sum_coords << std::endl;
    std::cout << "Difference in weights is: " << sum_weights << std::endl;

    delete w;
    delete coords;
}

