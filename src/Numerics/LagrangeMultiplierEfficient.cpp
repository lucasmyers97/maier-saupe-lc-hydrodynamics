#include "LagrangeMultiplierEfficient.hpp"

#include <deal.II/differentiation/ad.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include "Numerics/NumericalTools.hpp"

template <int order, int space_dim>
LagrangeMultiplierEfficient<order, space_dim>::
LagrangeMultiplierEfficient(double alpha, double tol, int max_iter)
    : inverted(false)
    , lmr(alpha, tol, max_iter)
    , ad_helper(msc::vec_dim<space_dim>, msc::vec_dim<space_dim>)

    , Q(msc::vec_dim<space_dim>)
    , Lambda(msc::vec_dim<space_dim>)
    , Jac(msc::vec_dim<space_dim>, msc::vec_dim<space_dim>)

    , Q_ad(msc::vec_dim<space_dim>)
    , q_pair(std::vector<ADNumberType>(msc::mat_dim<space_dim>), 0)
    , eig_dofs(msc::vec_dim<space_dim>)
    , Jac_input(msc::vec_dim<space_dim>, msc::vec_dim<space_dim>)

    , diag_dofs(msc::vec_dim<space_dim>)
    , Jac_reduced(msc::vec_dim<space_dim>, msc::vec_dim<space_dim>)
    , Lambda_dofs(msc::vec_dim<space_dim>)

    , Lambda_ad(msc::vec_dim<space_dim>)
    , Jac_output(msc::vec_dim<space_dim>, msc::vec_dim<space_dim>)
    , tmp(msc::vec_dim<space_dim>, msc::vec_dim<space_dim>)
{}



template <int order, int space_dim>
unsigned int LagrangeMultiplierEfficient<order, space_dim>::
invertQ(dealii::Vector<double> Q_in)
{
    bool force_auto_diff_perturbation = false;

    // read Q-values into a format that can be parsed
    for (unsigned int i = 0; i < msc::vec_dim<space_dim>; ++i)
    {
      Q[i] = Q_in[i];
      // Q[i] = (Q_in[i] == 0) ? 1e-18: Q_in[i];
      // if (Q[i] == 0) force_auto_diff_perturbation = true;
    }

    // get auto-differentiable numbers from ad_helper
    ad_helper.reset();
    ad_helper.register_independent_variables(Q);
    Q_ad = ad_helper.get_sensitive_variables();

    // turn Q-dofs into a matrix
    Q_mat[0][0] = Q_ad[0];
    Q_mat[0][1] = Q_ad[1];
    Q_mat[0][2] = Q_ad[2];
    Q_mat[1][1] = Q_ad[3];
    Q_mat[1][2] = Q_ad[4];
    Q_mat[2][2] = -(Q_ad[0] + Q_ad[3]);

    // Get rotation matrix which diagonalizes Q_mat
    eigs = dealii::eigenvectors(Q_mat,
                                dealii::SymmetricTensorEigenvectorMethod
                                // ::ql_implicit_shifts);
                                ::jacobi);
                                // ::hybrid);
                                // force_auto_diff_perturbation);
    for (unsigned int i = 0; i < msc::mat_dim<space_dim>; ++i)
        for (unsigned int j = 0; j < msc::mat_dim<space_dim>; ++j)
            R[i][j] = eigs[j].second[i];
    if (dealii::determinant(R) < 0)
        R *= -1;

    q_pair = NumericalTools::matrix_to_quaternion(R);

    // return the dofs (first part Lambda second part quaternions) for ad_helper
    eig_dofs[0] = eigs[0].first;
    eig_dofs[1] = eigs[1].first;
    eig_dofs[2] = q_pair.first[0];
    eig_dofs[3] = q_pair.first[1];
    eig_dofs[4] = q_pair.first[2];

    // register intermediate result, output first Jacobian, get values
    ad_helper.register_dependent_variables(eig_dofs);
    ad_helper.compute_jacobian(Jac_input);
    ad_helper.compute_values(diag_dofs);

    // invert Q for the diagonalized case
    Q_diag[0] = diag_dofs[0];
    Q_diag[1] = diag_dofs[1];
    unsigned int iters = lmr.invertQ(Q_diag);
    Lambda_diag = lmr.returnLambda();
    Jac_diag = lmr.returnJac();

    // get the middle Jacobian
    Jac_diag = dealii::invert(Jac_diag);
    Jac_reduced(0, 0) = Jac_diag[0][0];
    Jac_reduced(0, 1) = Jac_diag[0][1];
    Jac_reduced(1, 0) = Jac_diag[1][0];
    Jac_reduced(1, 1) = Jac_diag[1][1];
    Jac_reduced(2, 2) = 1;
    Jac_reduced(3, 3) = 1;
    Jac_reduced(4, 4) = 1;

    // package everything back up so it can be registered by ad_helper
    Lambda_dofs[0] = Lambda_diag[0];
    Lambda_dofs[1] = Lambda_diag[1];
    Lambda_dofs[2] = diag_dofs[2];
    Lambda_dofs[3] = diag_dofs[3];
    Lambda_dofs[4] = diag_dofs[4];

    // get out quaternions and diagonal Lambda values as autodifferentiable
    ad_helper.reset();
    ad_helper.register_independent_variables(Lambda_dofs);
    eig_dofs = ad_helper.get_sensitive_variables();
    q_pair.first[0] = eig_dofs[2];
    q_pair.first[1] = eig_dofs[3];
    q_pair.first[2] = eig_dofs[4];

    // rotate Lambda back into the original frame
    R = NumericalTools::quaternion_to_matrix(q_pair.first, q_pair.second);
    Lambda_mat[0][0] = eig_dofs[0];
    Lambda_mat[1][1] = eig_dofs[1];
    Lambda_mat[2][2] = -(eig_dofs[0] + eig_dofs[1]);
    Lambda_mat_full = R * Lambda_mat * dealii::transpose(R);

    // collect all the autodifferentiable numbers so ad_helper can compute
    Lambda_ad[0] = Lambda_mat_full[0][0];
    Lambda_ad[1] = Lambda_mat_full[0][1];
    Lambda_ad[2] = Lambda_mat_full[0][2];
    Lambda_ad[3] = Lambda_mat_full[1][1];
    Lambda_ad[4] = Lambda_mat_full[1][2];

    // compute Lambda and output Jacobian
    ad_helper.register_dependent_variables(Lambda_ad);
    ad_helper.compute_jacobian(Jac_output);
    ad_helper.compute_values(Lambda);

    // compose Jacobians to get Lambda Jac in original frame
    Jac_output.mmult(tmp, Jac_reduced);
    tmp.mmult(Jac, Jac_input);

    inverted = true;

    return iters;
}



template <int order, int space_dim>
double LagrangeMultiplierEfficient<order, space_dim>::returnZ() const
{
    assert(inverted && "Q not inverted in call to returnZ");
    return lmr.returnZ();
}



template <int order, int space_dim>
void LagrangeMultiplierEfficient<order, space_dim>::
returnLambda(dealii::Vector<double> &Lambda_out) const
{
    assert(inverted && "Q not inverted in call to returnLambda");
    Lambda_out = Lambda;
}



template <int order, int space_dim>
void LagrangeMultiplierEfficient<order, space_dim>::
returnJac(dealii::FullMatrix<double> &Jac_out) const
{
    assert(inverted && "Q not inverted in call to returnJac");
    Jac_out = Jac;
}

#include "LagrangeMultiplierEfficient.inst"
