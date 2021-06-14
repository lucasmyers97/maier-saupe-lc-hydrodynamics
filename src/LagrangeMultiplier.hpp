#ifndef LAGRANGEMULTIPLIER_HPP
#define LAGRANGEMULTIPLIER_HPP

#include <functional>
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
    static const Eigen::Map<
        Eigen::Matrix<double, order, mat_dim> > lebedev_coords;
    static const Eigen::Map<
        Eigen::Matrix<double, order, 1> > lebedev_weights;

    // damping coefficient for Newton's method
    const double alpha;

    static Eigen::Map<
        Eigen::Matrix<double, order, mat_dim> > makeLebedevCoords();
    static Eigen::Map<
        Eigen::Matrix<double, order, 1> > makeLebedevWeights();

    double sphereIntegral(
            std::function<double (double, double, double)> integrand);

public:
    LagrangeMultiplier(double in_alpha);
    void Test(
            std::function<double (double, double, double)> f);

};

#endif
