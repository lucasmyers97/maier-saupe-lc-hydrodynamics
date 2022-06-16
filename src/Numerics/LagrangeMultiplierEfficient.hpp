#ifndef LAGRANGE_MULTIPLIER_EFFICIENT_HPP
#define LAGRANGE_MULTIPLIER_EFFICIENT_HPP

#include <deal.II/base/tensor.h>
#include <deal.II/differentiation/ad.h>

#include <vector>
#include <array>
#include <type_traits>

#include "Utilities/maier_saupe_constants.hpp"
#include "Numerics/LagrangeMultiplierReduced.hpp"

namespace msc = maier_saupe_constants;

template <int order, int space_dim>
class LagrangeMultiplierEfficient
{
public:
    LagrangeMultiplierEfficient(double alpha_, double tol_, int max_iter_);

    unsigned int invertQ(dealii::Vector<double> Q_in);
    double returnZ() const;
    void returnLambda(dealii::Vector<double> &Lambda_out) const;
    void returnJac(dealii::FullMatrix<double> &Jac) const;

private:
    // set names of things
    static constexpr dealii::Differentiation::AD::NumberTypes ADTypeCode =
        dealii::Differentiation::AD::NumberTypes::sacado_dfad;
    using ADHelper =
        dealii::Differentiation::AD::VectorFunction<space_dim,
                                                    ADTypeCode,
                                                    double>;
    using ADNumberType = typename ADHelper::ad_type;

    bool inverted;

    std::vector<double> Q;
    dealii::Tensor<1, 2, double> Q_diag;
    dealii::Vector<double> Lambda;
    dealii::FullMatrix<double> Jac;

    std::vector<ADNumberType> Q_ad;
    dealii::SymmetricTensor<2, msc::mat_dim<space_dim>, ADNumberType> Q_mat;
    std::array<std::pair<ADNumberType,
                         dealii::Tensor<1, msc::mat_dim<space_dim>,ADNumberType>
                         >,
               std::integral_constant<int, msc::mat_dim<space_dim>>::value>
    eigs;
    dealii::Tensor<2, msc::mat_dim<space_dim>, ADNumberType> R;
    std::pair<std::vector<ADNumberType>, unsigned int> q_pair;
    std::vector<ADNumberType> eig_dofs;
    dealii::FullMatrix<double> Jac_input;
    dealii::Vector<double> diag_dofs;

    dealii::Tensor<1, 2, double> Lambda_diag;
    dealii::Tensor<2, 2, double> Jac_diag;
    dealii::FullMatrix<double> Jac_reduced;
    std::vector<double> Lambda_dofs;

    dealii::SymmetricTensor<2, msc::mat_dim<space_dim>, ADNumberType>
        Lambda_mat;
    dealii::Tensor<2, msc::mat_dim<space_dim>, ADNumberType> Lambda_mat_full;
    std::vector<ADNumberType> Lambda_ad;
    dealii::FullMatrix<double> Jac_output;
    dealii::FullMatrix<double> tmp;

    ADHelper ad_helper;
    LagrangeMultiplierReduced lmr;
};

#endif
