#ifndef LAGRANGE_MULTIPLIER_ANALYTIC_HPP
#define LAGRANGE_MULTIPLIER_ANALYTIC_HPP

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <boost/serialization/access.hpp>
#include <boost/serialization/serialization.hpp>

#include <vector>

#include "Numerics/LagrangeMultiplierReduced.hpp"
#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
class LagrangeMultiplierAnalytic
{

public:
    LagrangeMultiplierAnalytic(const int orderi=590, 
                               const double alpha=1.0,
                               const double tol=1e-10, 
                               const int max_iter=20,
                               const double degnerate_tol=1e-8);

    void invertQ(const dealii::Vector<double> &Q_in);
    double returnZ() const;
    void returnLambda(dealii::Vector<double> &Lambda_out) const;
    void returnJac(dealii::FullMatrix<double> &Jac_out) const;

private:

    void diagonalizeQ();
    void invertReducedQ();
    void undiagonalizeLambda();
    void calcJacobian();

    double degenerate_tol;

    LagrangeMultiplierReduced lmr;

    dealii::SymmetricTensor<2, msc::mat_dim<dim>, double> Q_mat;
    dealii::Vector<double> Lambda;
    dealii::FullMatrix<double> Jac;
    double Z;

    dealii::Tensor<2, msc::mat_dim<dim>, double> R;
    bool degenerate_eigenvalues;
    dealii::Tensor<1, 2, double> Q_red;
    dealii::FullMatrix<double> dlambda;
    std::vector<dealii::FullMatrix<double>> S;

    dealii::Tensor<1, 2, double> Lambda_red;
    dealii::Tensor<2, 2, double> Jac_red;
    dealii::FullMatrix<double> dLambda;

    dealii::SymmetricTensor<2, msc::mat_dim<dim>, double> Lambda_mat;
    dealii::Tensor<2, msc::mat_dim<dim>, double> Lambda_mat_full;
    std::vector<dealii::FullMatrix<double>> T;
    dealii::FullMatrix<double> dF;
    std::vector<double> gamma;

    std::vector<dealii::FullMatrix<double>> TS;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & degenerate_tol;
        ar & lmr;
    }
};


#endif
