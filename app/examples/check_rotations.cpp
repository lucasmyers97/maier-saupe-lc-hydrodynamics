#include <deal.II/differentiation/ad.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <vector>
#include <cmath>
#include <iostream>

int main()
{
    const int dim = 3;
    const int vec_dim = 5;

    dealii::Vector<double> Q_vec(vec_dim);
    Q_vec[0] = 0.15;
    Q_vec[1] = 0.25;
    Q_vec[2] = 0.05;
    Q_vec[3] = 0.15;
    Q_vec[4] = 0.0;

    // diagonalize and keep track of eigen-numbers
    dealii::SymmetricTensor<2, dim, double> Q;
    Q[0][0] = Q_vec[0];
    Q[0][1] = Q_vec[1];
    Q[0][2] = Q_vec[2];
    Q[1][1] = Q_vec[3];
    Q[1][2] = Q_vec[4];
    Q[2][2] = -(Q_vec[0] + Q_vec[3]);
    auto eigs = dealii::eigenvectors(Q);

    dealii::Tensor<2, dim, double> R;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            R[i][j] = eigs[j].second[i];

    std::cout << "Printing determinant " << dealii::determinant(R) << std::endl;

    R = -R;

    std::cout << std::endl << "Printing Q" << std::endl;
    std::cout << Q << std::endl;

    std::cout << std::endl << "Printing rotation" << std::endl;
    std::cout << R << std::endl << std::endl;

    std::cout << std::endl << "Printing diagonalized Q" << std::endl;
    std::cout << dealii::transpose(R) * Q * R << std::endl << std::endl;

    std::cout << std::endl << "Printing undiagonalized Q" << std::endl;
    std::cout << R * dealii::transpose(R) * Q * R * dealii::transpose(R) << std::endl << std::endl;

    // double qk = 0.5 * std::sqrt(1 - R[0][0] - R[1][1] + R[2][2]);
    // double qi = (R[0][2] + R[2][0]) / (4 * qk);
    // double qj = (R[2][1] + R[1][2]) / (4 * qk);
    // double qr = (R[1][0] - R[0][1]) / (4 * qk);

    double qr = 0.5 * std::sqrt(1 + R[0][0] + R[1][1] + R[2][2]);
    double qi = (R[2][1] - R[1][2]) / (4 * qr);
    double qj = (R[0][2] - R[2][0]) / (4 * qr);
    double qk = (R[1][0] - R[0][1]) / (4 * qr);

    std::cout << std::endl << "Printing quaternions "
              << qi << " " << qj << " " << qk << " " << qr << std::endl;

    double qi2 = 0.5 * std::sqrt(1 + R[0][0] - R[1][1] - R[2][3]);

    std::cout << "Printing q's " << qi << " " << qi2 << std::endl;

    R[0][0] = 1 - 2*qj*qj - 2*qk*qk;
    R[0][1] = 2 * (qi*qj - qk*qr);
    R[0][2] = 2 * (qi*qk + qj*qr);
    R[1][0] = 2 * (qi*qj + qk*qr);
    R[1][1] = 1 - 2*qi*qi - 2*qk*qk;
    R[1][2] = 2 * (qj*qk - qi*qr);
    R[2][0] = 2 * (qi*qk - qj*qr);
    R[2][1] = 2 * (qj*qk + qi*qr);
    R[2][2] = 1 - 2*qi*qi - 2*qj*qj;

    // R[0][0] = 2*qi*qi + 2*qr*qr - 1;
    // R[0][1] = 2 * (qi*qj - qk*qr);
    // R[0][2] = 2 * (qi*qk + qj*qr);
    // R[1][0] = 2 * (qi*qj + qk*qr);
    // R[1][1] = 2*qj*qj + 2*qr*qr - 1;
    // R[1][2] = 2 * (qj*qk - qi*qr);
    // R[2][0] = 2 * (qi*qk - qj*qr);
    // R[2][1] = 2 * (qj*qk + qi*qr);
    // R[2][2] = 2*qk*qk + 2*qr*qr - 1;

    std::cout << R << std::endl << std::endl;

    std::cout << dealii::transpose(R) * Q * R << std::endl;

    return 0;
}
