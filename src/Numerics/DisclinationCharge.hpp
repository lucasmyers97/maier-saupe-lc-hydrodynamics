#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>

#include <vector>

namespace NumericalTools
{
    template <int dim>
    inline void 
    DisclinationCharge(const std::vector<dealii::Tensor<1, dim, double>> &dQ, 
                       dealii::Tensor<2, 3, double> &D)
    {
        D[0][0] = 2 * (       dQ[0][1]*dQ[4][2] - dQ[0][2]*dQ[4][1]
                       +      dQ[1][1]*dQ[2][2] - dQ[1][2]*dQ[2][1]
                       + 2 * (dQ[3][1]*dQ[4][2] - dQ[3][2]*dQ[4][1]));
        D[1][0] = 2 * (       dQ[2][1]*dQ[3][2] - dQ[2][2]*dQ[3][1]
                       +      dQ[1][2]*dQ[4][1] - dQ[1][1]*dQ[4][2]
                       + 2 * (dQ[0][2]*dQ[2][1] - dQ[0][1]*dQ[2][2]));
        D[2][0] = 2 * (       dQ[0][1]*dQ[1][2] - dQ[0][2]*dQ[1][1]
                       +      dQ[1][1]*dQ[3][2] - dQ[1][2]*dQ[3][1]
                       +      dQ[2][1]*dQ[4][2] - dQ[2][2]*dQ[4][1]);
        D[0][1] = 2 * (       dQ[0][2]*dQ[4][0] - dQ[0][0]*dQ[4][2]
                       +      dQ[1][2]*dQ[2][0] - dQ[1][0]*dQ[2][2]
                       + 2 * (dQ[3][2]*dQ[4][0] - dQ[3][0]*dQ[4][2]));
        D[1][1] = 2 * (       dQ[2][2]*dQ[3][0] - dQ[2][0]*dQ[3][2]
                       +      dQ[1][0]*dQ[4][2] - dQ[1][2]*dQ[4][0]
                       + 2 * (dQ[0][0]*dQ[2][2] - dQ[0][2]*dQ[2][0]));
        D[2][1] = 2 * (       dQ[0][2]*dQ[1][0] - dQ[0][0]*dQ[1][2]
                       +      dQ[1][2]*dQ[3][0] - dQ[1][0]*dQ[3][2]
                       +      dQ[2][2]*dQ[4][0] - dQ[2][0]*dQ[4][2]);
        D[0][2] = 2 * (       dQ[0][0]*dQ[4][1] - dQ[0][1]*dQ[4][0]
                       +      dQ[1][0]*dQ[2][1] - dQ[1][1]*dQ[2][0]
                       + 2 * (dQ[3][0]*dQ[4][1] - dQ[3][1]*dQ[4][0]));
        D[1][2] = 2 * (       dQ[2][0]*dQ[3][1] - dQ[2][1]*dQ[3][0]
                       +      dQ[1][1]*dQ[4][0] - dQ[1][0]*dQ[4][1]
                       + 2 * (dQ[0][1]*dQ[2][0] - dQ[0][0]*dQ[2][1]));
        D[2][2] = 2 * (       dQ[0][0]*dQ[1][1] - dQ[0][1]*dQ[1][0]
                       +      dQ[1][0]*dQ[3][1] - dQ[1][1]*dQ[3][0]
                       +      dQ[2][0]*dQ[4][1] - dQ[2][1]*dQ[4][0]);
    }
} // namespace NumericalTools
