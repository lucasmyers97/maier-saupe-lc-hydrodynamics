#ifndef LAGRANGEMULTIPLIER_HPP
#define LAGRANGEMULTIPLIER_HPP

#include <functional>
#include <vector>
#include <array>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/function.h>

template <int order>
class LagrangeMultiplier
{
public:
    LagrangeMultiplier(double in_alpha,
            		   double in_tol,
					   unsigned int in_max_iter);

    void setQ(dealii::Vector<double> &new_Q);
    void returnLambda(dealii::Vector<double> &outLambda);
    void returnJac(dealii::LAPACKFullMatrix<double> &outJac);

    void lagrangeTest();

private:
    // problem dimensions
    static constexpr int vec_dim = 5;
    static constexpr int mat_dim = 3;

    // For implementing Newton's method
    unsigned int invertQ();
    void updateRes();
    void updateJac();
    void updateVariation();

    // For Lebedev Quadrature
    double sphereIntegral(
                    std::function<double (dealii::Point<mat_dim>)> integrand);
    static std::vector<dealii::Point<mat_dim>> makeLebedevCoords();
    static std::vector<double> makeLebedevWeights();

    // Helper for doing xi_i \Lambda_{ij} xi_j sum
    double lambdaSum(dealii::Point<mat_dim> x);

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

    // (i, j) indices in Q-tensor given Q-vector index
    static constexpr std::array<int, vec_dim> Q_row = {0, 0, 0, 1, 1};
    static constexpr std::array<int, vec_dim> Q_col = {0, 1, 2, 1, 2};

    // Q-vector indices given Q-tensor (i, j) indices (except (3, 3) entry)
    static constexpr std::array<std::array<int, mat_dim>, mat_dim>
        Q_idx = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 0}}};

    // Kronecker delta
    static constexpr std::array<std::array<int, mat_dim>, mat_dim>
        delta = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

    // Points on sphere + weights for Lebedev Quadrature
    static const std::vector<dealii::Point<mat_dim>> lebedev_coords;
    static const std::vector<double> lebedev_weights;

    void printVecTest(std::function<double (dealii::Point<mat_dim>)> integrand);
};

#endif
