#ifndef CALC_S_VALUE_HPP
#define CALC_S_VALUE_HPP

#include <deal.II/base/symmetric_tensor.h>

#include "Utilities/maier_saupe_constants.hpp"

namespace NumericalTools
{
    namespace msc = maier_saupe_constants;

    template <int dim>
    inline double
    calcSValue(dealii::SymmetricTensor<2, msc::mat_dim<dim>, double> &Q)
    {
        auto eigs 
            = dealii::eigenvectors(Q, dealii::SymmetricTensorEigenvectorMethod
                                      ::jacobi); 

        // find index of maximal eigenvalue
        std::size_t max_eig_idx = 0;
        for (unsigned int i = 0; i < msc::mat_dim<dim>; ++i)
            if (eigs[i].first > eigs[max_eig_idx].first)
                max_eig_idx = i;

        return eigs[max_eig_idx].first;
    }

} // namespace NumericalTools

#endif
