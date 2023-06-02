#include "LagrangeMultiplierAnalytic.hpp"

#include <deal.II/lac/vector.h>

#include <cmath>

#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
LagrangeMultiplierAnalytic<dim>::
LagrangeMultiplierAnalytic(const int order, const double alpha,
                           const double tol, const int max_iter,
                           const double degenerate_tol)
    : lmr(order, alpha, tol, max_iter)
    , degenerate_tol(degenerate_tol)

    , Lambda(msc::vec_dim<dim>)
    , Jac(msc::vec_dim<dim>, msc::vec_dim<dim>)

    , dlambda(2, msc::vec_dim<dim>)
    , S(3, dealii::FullMatrix<double>(msc::mat_dim<dim>, msc::vec_dim<dim>))

    , dLambda(2, 2)

    , T(2, dealii::FullMatrix<double>(msc::vec_dim<dim>, msc::mat_dim<dim>))
    , dF(msc::vec_dim<dim>, 2)
    , gamma(3)

    , TS(3, dealii::FullMatrix<double>(msc::vec_dim<dim>, msc::vec_dim<dim>))
{}



template <int dim>
void LagrangeMultiplierAnalytic<dim>::
    invertQ(const dealii::Vector<double> &Q_in)
{
    Q_mat[0][0] = Q_in[0];
    Q_mat[0][1] = Q_in[1];
    Q_mat[0][2] = Q_in[2];
    Q_mat[1][1] = Q_in[3];
    Q_mat[1][2] = Q_in[4];
    Q_mat[2][2] = -(Q_in[0] + Q_in[3]);

    diagonalizeQ();
    invertReducedQ();
    undiagonalizeLambda();
    calcJacobian();
}



template <int dim>
double LagrangeMultiplierAnalytic<dim>::returnZ() const
{
    return lmr.returnZ();
}



template <int dim>
void LagrangeMultiplierAnalytic<dim>::
    returnLambda(dealii::Vector<double> &Lambda_out) const
{
    Lambda_out = Lambda;
}



template <int dim>
void LagrangeMultiplierAnalytic<dim>::
    returnJac(dealii::FullMatrix<double> &Jac_out) const
{
  Jac_out = Jac;
}



template <int dim>
void LagrangeMultiplierAnalytic<dim>::diagonalizeQ()
{
    auto eigs = dealii::eigenvectors(Q_mat);

    double eps_1 = std::abs(eigs[0].first - eigs[1].first);
    double eps_2 = std::abs(eigs[1].first - eigs[2].first);
    double eps = 0;

    if (eps_1 < eps_2)
    {
        eps = eps_1;
        Q_red[0] = eigs[0].first;
        for (unsigned int i = 0; i < msc::mat_dim<dim>; ++i)
        {
          R[i][0] = eigs[0].second[i];
          R[i][2] = eigs[2].second[i];
        }
    } else
    {
        eps = eps_2;
        Q_red[0] = eigs[2].first;
        for (unsigned int i = 0; i < msc::mat_dim<dim>; ++i)
        {
            R[i][0] = eigs[2].second[i];
            R[i][2] = eigs[0].second[i];
        }
    }

    Q_red[1] = eigs[1].first;
    for (unsigned int i = 0; i < msc::mat_dim<dim>; ++i)
        R[i][1] = eigs[1].second[i];

    if (dealii::determinant(R) < 0)
      R *= -1;

    degenerate_eigenvalues = (eps < degenerate_tol);

    for (unsigned int i = 0; i < 2; ++i)
    {
        dlambda(i, 0) = R[0][i] * R[0][i] - R[2][i] * R[2][i];
        dlambda(i, 1) = R[0][i] * R[1][i] + R[1][i] * R[0][i];
        dlambda(i, 2) = R[0][i] * R[2][i] + R[2][i] * R[0][i];
        dlambda(i, 3) = R[1][i] * R[1][i] - R[2][i] * R[2][i];
        dlambda(i, 4) = R[1][i] * R[2][i] + R[2][i] * R[1][i];
    }

    std::vector<double> dB_ni_nj(msc::vec_dim<dim>);
    for (unsigned int l = 0; l < 3; ++l)
    {
        int i = (l < 2) ? 0 : 1;
        int j = (l < 1) ? 1 : 2;

        dB_ni_nj[0] = R[0][i] * R[0][j] - R[2][i] * R[2][j];
        dB_ni_nj[1] = R[0][i] * R[1][j] + R[1][i] * R[0][j];
        dB_ni_nj[2] = R[0][i] * R[2][j] + R[2][i] * R[0][j];
        dB_ni_nj[3] = R[1][i] * R[1][j] - R[2][i] * R[2][j];
        dB_ni_nj[4] = R[1][i] * R[2][j] + R[2][i] * R[1][j];

        for (unsigned int k = 0; k < dim; ++k)
            for (unsigned int m = 0; m < msc::vec_dim<dim>; ++m)
                S[l](k, m) = R[k][j] * dB_ni_nj[m];
    }
}



template <int dim>
void LagrangeMultiplierAnalytic<dim>::invertReducedQ()
{
    lmr.invertQ(Q_red);
    Z = lmr.returnZ();
    Lambda_red = lmr.returnLambda();
    Jac_red = lmr.returnJac();
    Jac_red = dealii::invert(Jac_red);

    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        dLambda(i, j) = Jac_red[i][j];
}



template <int dim>
void LagrangeMultiplierAnalytic<dim>::undiagonalizeLambda()
{
    Lambda_mat[0][0] = Lambda_red[0];
    Lambda_mat[1][1] = Lambda_red[1];
    Lambda_mat[2][2] = -(Lambda_red[0] + Lambda_red[1]);

    Lambda_mat_full = R * Lambda_mat * dealii::transpose(R);

    Lambda[0] = Lambda_mat_full[0][0];
    Lambda[1] = Lambda_mat_full[0][1];
    Lambda[2] = Lambda_mat_full[0][2];
    Lambda[3] = Lambda_mat_full[1][1];
    Lambda[4] = Lambda_mat_full[1][2];

    for (unsigned int i = 0; i < 2; ++i)
    {
        T[i](0, 0) = 2 * R[0][i];
        T[i](1, 0) = R[1][i];
        T[i](2, 0) = R[2][i];
        T[i](1, 1) = R[0][i];
        T[i](3, 1) = 2 * R[1][i];
        T[i](4, 1) = R[2][i];
        T[i](2, 2) = R[0][i];
        T[i](4, 2) = R[1][i];
    }
    for (unsigned int i = 0; i < 2; ++i)
    {
        dF(0, i) = R[0][i] * R[0][i] - R[0][2] * R[0][2];
        dF(1, i) = R[0][i] * R[1][i] - R[0][2] * R[1][2];
        dF(2, i) = R[0][i] * R[2][i] - R[0][2] * R[2][2];
        dF(3, i) = R[1][i] * R[1][i] - R[1][2] * R[1][2];
        dF(4, i) = R[1][i] * R[2][i] - R[1][2] * R[2][2];
    }
}



template <int dim>
void LagrangeMultiplierAnalytic<dim>::calcJacobian()
{
    Jac = 0;
    Jac.triple_product(dLambda, dF, dlambda);

    T[0].mmult(TS[0], S[0]);
    T[0].mmult(TS[1], S[1]);
    T[1].mmult(TS[2], S[2]);

    gamma[1] = 1 / (2*Q_red[0] + Q_red[1]);
    gamma[2] = 1 / (Q_red[0] + 2*Q_red[1]);

    if (degenerate_eigenvalues)
    {
        Jac.add((dLambda(0, 0) - dLambda(0, 1)), TS[0],
                3 * Lambda_red[0] * gamma[1], TS[1],
                3 * Lambda_red[0] * gamma[2], TS[2]);
    } else
    {
        gamma[0] = 1 / (Q_red[0] - Q_red[1]);

        Jac.add((Lambda_red[0] - Lambda_red[1]) * gamma[0], TS[0],
                (2 * Lambda_red[0] + Lambda_red[1]) * gamma[1], TS[1],
                (Lambda_red[0] + 2 * Lambda_red[1]) * gamma[2], TS[2]);
    }
}

template class LagrangeMultiplierAnalytic<2>;
template class LagrangeMultiplierAnalytic<3>;

