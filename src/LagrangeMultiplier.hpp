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

    unsigned int invertQ(dealii::Vector<double> &Q_in);
    void returnLambda(dealii::Vector<double> &outLambda);
    void returnJac(dealii::LAPACKFullMatrix<double> &outJac);

    // void lagrangeTest();

private:
    // For implementing Newton's method
    void initializeInversion(dealii::Vector<double> &Q_in);
    void updateResJac();
    void updateVariation();

    double calcInt1Term(const double exp_lambda, 
                        const int quad_idx, const int i_m, const int j_m);
    double calcInt2Term(const double exp_lambda, 
                        const int quad_idx, const int i_m, const int j_m, 
                        const int i_n, const int j_n);
    double calcInt3Term(const double exp_lambda,
                        const int quad_idx,
                        const int i_m, const int j_m,
                        const int i_n, const int j_n);
    double calcInt4Term(const double exp_lambda,
                        const int quad_idx, const int i_m, const int j_m);

    // For Lebedev Quadrature
    static std::vector<dealii::Point<maier_saupe_constants::mat_dim<space_dim>>> 
        makeLebedevCoords();
    static std::vector<double> makeLebedevWeights();

    // Helper for doing xi_i \Lambda_{ij} xi_j sum
    double lambdaSum(dealii::Point<maier_saupe_constants::mat_dim<space_dim>> x);

    // Flags
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
    double Z{};

    // Arrays for storing integrals
    using int_vec 
        = std::array<double, maier_saupe_constants::vec_dim<space_dim>>;
    using int_mat
        = std::array<int_vec, maier_saupe_constants::vec_dim<space_dim>>; 
    int_vec int1 = {0};
    int_mat int2 = {{0}};
    int_mat int3 = {{0}};
    int_vec int4 = {0};

    // Points on sphere + weights for Lebedev Quadrature
    static const std::vector<dealii::Point<maier_saupe_constants::mat_dim<space_dim>>> lebedev_coords;
    static const std::vector<double> lebedev_weights;

    // void printVecTest(std::function<double (dealii::Point<maier_saupe_constants::mat_dim<space_dim>>)> integrand);
};

#endif