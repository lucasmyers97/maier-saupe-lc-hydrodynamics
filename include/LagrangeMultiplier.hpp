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
    static const dealii::Vector<double> x;
    static const dealii::Vector<double> y;
    static const dealii::Vector<double> z;
    static const dealii::Vector<double> w;

    // damping coefficient for Newton's method
    const double alpha;

    static dealii::Vector<double> makeLebedev(char c);

    double sphereIntegral(
            std::function<double (double, double, double)> integrand);

public:
    LagrangeMultiplier(double in_alpha);
    void printVecTest();
//     void Test(
//             std::function<double (double, double, double)> f);

};

#endif
