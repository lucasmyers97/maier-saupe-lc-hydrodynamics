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
    static constexpr std::array<int, vec_dim> Q_row = {0, 0, 0, 1, 1};
    static constexpr std::array<int, vec_dim> Q_col = {0, 1, 2, 1, 2};
    static constexpr std::array<std::array<int, mat_dim>, mat_dim>
        Q_idx = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 0}}};
    static constexpr std::array<std::array<int, mat_dim>, mat_dim>
        delta = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

    // order
    static constexpr int order = 2702;

    static const std::vector<dealii::Point<mat_dim>> lebedev_coords;
    static const std::vector<double> lebedev_weights;

    dealii::Vector<double> Q;
    dealii::Vector<double> Lambda;

    // damping coefficient for Newton's method
    const double alpha;

    static std::vector<dealii::Point<mat_dim>> makeLebedevCoords();
    static std::vector<double> makeLebedevWeights();

    double sphereIntegral(
            std::function<double (dealii::Point<mat_dim>)> integrand);

public:
    dealii::Vector<double> Res;
    dealii::LAPACKFullMatrix<double> Jac;
    dealii::Vector<double> dLambda;
    LagrangeMultiplier(double in_alpha);
    void updateRes();
    void updateJac();
    void updateVariation();
    double calcLagrangeExp(dealii::Point<mat_dim> x);
    void printVecTest(std::function<double (dealii::Point<mat_dim>)> integrand);
};

#endif
