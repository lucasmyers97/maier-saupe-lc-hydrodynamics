#ifndef LAGRANGEMULTIPLIER_HPP
#define LAGRANGEMULTIPLIER_HPP

#include <functional>
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
    static constexpr int i[vec_dim] = {0, 0, 0, 1, 1};
    static constexpr int j[vec_dim] = {0, 1, 2, 1, 2};

    // order + coords + weights of lebedev quadrature
    static constexpr int order = 2702;

    // need to replace these with dealII vectors
    static const dealii::FullMatrix<double> lebedev_coords;
    static const dealii::Vector<double> lebedev_weights;

    // damping coefficient for Newton's method
    const double alpha;

    // replace these with dealII vectors
    static dealii::FullMatrix<double> makeLebedevCoords();
    static dealii::Vector<double> makeLebedevWeights();

    double sphereIntegral(
            std::function<double (double, double, double)> integrand);

public:
    LagrangeMultiplier(double in_alpha);
    void printVecTest();
//     void Test(
//             std::function<double (double, double, double)> f);

};

#endif
