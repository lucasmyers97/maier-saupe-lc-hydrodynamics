#ifndef NUMERICAL_TOOLS_HPP
#define NUMERICAL_TOOLS_HPP

#include <deal.II/base/tensor.h>

#include <utility>
#include <vector>
#include <cmath>

namespace NumericalTools
{
    template <typename Number>
    std::pair<std::vector<Number>, unsigned int>
    matrix_to_quaternion(const dealii::Tensor<2, 3, Number> &R)
    {
        constexpr int n_quaternion_dofs = 3;
        std::vector<Number> q(n_quaternion_dofs);
        unsigned int first_q_calculated = 0;

        if ((R[0][0] >= 0) && (R[1][1] >=0))
        {
            // calculate real first
            first_q_calculated = 0;
            Number qr = 0.5 * std::sqrt(1 + R[0][0] + R[1][1] + R[2][2]);
            q[0] = (R[2][1] - R[1][2]) / (4 * qr); // qi
            q[1] = (R[0][2] - R[2][0]) / (4 * qr); // qj
            q[2] = (R[1][0] - R[0][1]) / (4 * qr); // qk
        }
        else if ((R[0][0] >= 0) && (R[1][1] < 0))
        {
            // calculate i first
            first_q_calculated = 1;
            Number qi = 0.5 * std::sqrt(1 + R[0][0] - R[1][1] - R[2][2]);
            q[0] = (R[2][1] - R[1][2]) / (4 * qi); // qr
            q[1] = (R[1][0] + R[0][1]) / (4 * qi); // qj
            q[2] = (R[0][2] + R[2][0]) / (4 * qi); // qk
        }
        else if ((R[0][0] < 0) && (R[1][1] >= 0))
        {
            // calculate j first
            first_q_calculated = 2;
            Number qj = 0.5 * std::sqrt(1 - R[0][0] + R[1][1] - R[2][2]);
            q[0] = (R[0][2] - R[2][0]) / (4 * qj); // qr
            q[1] = (R[1][0] + R[0][1]) / (4 * qj); // qi
            q[2] = (R[2][1] + R[1][2]) / (4 * qj); // qk
        }
        else
        {
            // calculate k first
            first_q_calculated = 3;
            Number qk = 0.5 * std::sqrt(1 - R[0][0] - R[1][1] + R[2][2]);
            q[0] = (R[1][0] - R[0][1]) / (4 * qk); // qr
            q[1] = (R[0][2] + R[2][0]) / (4 * qk); // qi
            q[2] = (R[2][1] + R[1][2]) / (4 * qk); // qj
        }

        return std::make_pair(q, first_q_calculated);
    }



    template <typename Number>
    dealii::Tensor<2, 3, Number>
    quaternion_to_matrix(std::vector<Number> q, unsigned int first_q_calculated)
    {
        Number qr;
        Number qi;
        Number qj;
        Number qk;

        switch (first_q_calculated)
        {
        case 0:
            qi = q[0];
            qj = q[1];
            qk = q[2];
            qr = std::sqrt(1 - (qi*qi + qj*qj + qk*qk));
            break;
        case 1:
            qr = q[0];
            qj = q[1];
            qk = q[2];
            qi = std::sqrt(1 - (qr * qr + qj * qj + qk * qk));
            break;
        case 2:
            qr = q[0];
            qi = q[1];
            qk = q[2];
            qj = std::sqrt(1 - (qr*qr + qi*qi + qk*qk));
            break;
        case 3:
            qr = q[0];
            qi = q[1];
            qj = q[2];
            qk = std::sqrt(1 - (qr*qr + qi*qi + qj*qj));
            break;
        }

        dealii::Tensor<2, 3, Number> R;

        R[0][0] = 1 - 2 * (qj*qj + qk*qk);
        R[0][1] = 2 * (qi*qj - qk*qr);
        R[0][2] = 2 * (qi*qk + qj*qr);
        R[1][0] = 2 * (qi*qj + qk*qr);
        R[1][1] = 1 - 2 * (qi*qi + qk*qk);
        R[1][2] = 2 * (qj*qk - qi*qr);
        R[2][0] = 2 * (qi*qk - qj*qr);
        R[2][1] = 2 * (qj*qk + qi*qr);
        R[2][2] = 1 - 2 * (qi*qi + qj*qj);

        return R;
    }



    std::vector<double> linspace(double begin, double end, unsigned int num)
    {
        std::vector<double> range(num);
        double dx = (end - begin) / (num - 1);
        for (std::size_t i = 0; i < range.size(); ++i)
            range[i] = begin + i*dx;

        return range;
    }
} // namespace NumericalTools

#endif
