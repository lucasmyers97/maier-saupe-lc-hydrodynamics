#ifndef LAGRANGEMULTIPLIER_H
#define LAGRANGEMULTIPLIER_H

#include <functional>
#include <eigen3/Eigen/Dense>
 
class LagrangeMultiplier
{
private:
    static constexpr int vec_dim = 5;
    static constexpr int mat_dim = 3;
    const double alpha;
    Eigen::Matrix<int, vec_dim, 1> i;
    Eigen::Matrix<int, vec_dim, 1> j;

public:
    LagrangeMultiplier(double in_alpha);
    void Test();
    double sphereIntegral(int order,
            std::function<double (double, double, double)> integrand);

};

#endif
