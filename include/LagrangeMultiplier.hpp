#ifndef LAGRANGEMULTIPLIER_HPP
#define LAGRANGEMULTIPLIER_HPP

#include <functional>
#include <vector>
#include <array>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <eigen3/Eigen/Dense>
 
class LagrangeMultiplier
{
private:
    // problem dimensions
    static constexpr int vec_dim = 5;
    static constexpr int mat_dim = 3;

    // (i, j) coords in Q-tensor for Q1, Q2, Q3, Q4, Q5
    static constexpr std::array<int, vec_dim> i = {0, 0, 0, 1, 1};
    static constexpr std::array<int, vec_dim> j = {0, 1, 2, 1, 2};

    // order
    static constexpr int order = 2702;

    static const std::vector<dealii::Point<mat_dim>> lebedev_coords;
    static const std::vector<double> lebedev_weights;

    dealii::Vector<double> Lambda;
    dealii::Vector<double> dLambda;
    dealii::Tensor<2, vec_dim> Jac;
    dealii::Vector<double> Res;

    // damping coefficient for Newton's method
    const double alpha;

    static std::vector<dealii::Point<mat_dim>> makeLebedevCoords();
    static std::vector<double> makeLebedevWeights();

    double sphereIntegral(
            std::function<double (double, double, double)> integrand);

public:
    LagrangeMultiplier(double in_alpha);
    void printVecTest();
};

#endif
