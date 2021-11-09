#ifndef LAGRANGEMULTIPLIER_HPP
#define LAGRANGEMULTIPLIER_HPP

#include <functional>
#include <vector>
#include <array>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/function.h>
#include <maier_saupe_constants.hpp>

template <int order, int space_dim = 3>
class LagrangeMultiplier
{
public:
    LagrangeMultiplier(double alpha_,
            		   double tol_,
					   unsigned int max_iter_);

    void setQ(dealii::Vector<double> &new_Q);
    void returnLambda(dealii::Vector<double> &outLambda);
    void returnJac(dealii::LAPACKFullMatrix<double> &outJac);

    double calcZ();

    void lagrangeTest();

private:
    // For implementing Newton's method
    unsigned int invertQ();
    void updateRes();
    void updateJac();
    void updateVariation();

    // For Lebedev Quadrature
    double sphereIntegral
        (std::function<double (dealii::Point<maier_saupe_constants::mat_dim<space_dim>>)> integrand);
    static std::vector<dealii::Point<maier_saupe_constants::mat_dim<space_dim>>> makeLebedevCoords();
    static std::vector<double> makeLebedevWeights();

    // Helper for doing xi_i \Lambda_{ij} xi_j sum
    double lambdaSum(dealii::Point<maier_saupe_constants::mat_dim<space_dim>> x);

    // Flags
    bool Q_set;
    bool inverted;
    bool Jac_updated;

    // Vector to invert, and inversion solution
    dealii::Vector<double> Q;
    dealii::Vector<double> Lambda;

    // Newton's method parameters
    const double alpha;
    double tol;
    unsigned int max_iter;

    // Interim variables for Newton's method
    dealii::Vector<double> Res;
    dealii::LAPACKFullMatrix<double> Jac;
    dealii::Vector<double> dLambda;


    // Points on sphere + weights for Lebedev Quadrature
    static const std::vector<dealii::Point<maier_saupe_constants::mat_dim<space_dim>>> lebedev_coords;
    static const std::vector<double> lebedev_weights;

    void printVecTest(std::function<double (dealii::Point<maier_saupe_constants::mat_dim<space_dim>>)> integrand);
};

#endif