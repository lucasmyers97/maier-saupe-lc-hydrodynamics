#ifndef NEMATIC_POSTPROCESSOR_HPP
#define NEMATIC_POSTPROCESSOR_HPP

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <vector>
#include <string>
#include <array>
#include <utility>
#include <type_traits>

#include "Utilities/maier_saupe_constants.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
class NematicPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    NematicPostprocessor() {};

    inline virtual void evaluate_vector_field(
        const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<dealii::Vector<double>> &computed_quantities) const override;

    inline virtual std::vector<std::string> get_names() const override;

    inline virtual std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    inline virtual dealii::UpdateFlags get_needed_update_flags() const override;
};



template <int dim>
dealii::UpdateFlags NematicPostprocessor<dim>::get_needed_update_flags() const
{
    return dealii::update_values | dealii::update_quadrature_points;
}



template <int dim>
std::vector<std::string> NematicPostprocessor<dim>::get_names() const
{
    std::vector<std::string> solution_names(dim, "director");
    solution_names.emplace_back("S");

    return solution_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
NematicPostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim,
                   dealii::DataComponentInterpretation
                   ::component_is_part_of_vector);
    interpretation.push_back(dealii::DataComponentInterpretation
                             ::component_is_scalar);

    return interpretation;
}




template <int dim>
inline void NematicPostprocessor<dim>::evaluate_vector_field
(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == msc::vec_dim<dim>,
           ExcInternalError());

    dealii::SymmetricTensor<2, msc::mat_dim<dim>> Q;
    std::array<std::pair<double, dealii::Tensor<1, msc::mat_dim<dim>, double>>,
               std::integral_constant<int, msc::mat_dim<dim>>::value> eigs;
    unsigned int max_eig_idx = 0;
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        Q[0][0] = inputs.solution_values[q](0);
        Q[0][1] = inputs.solution_values[q](1);
        Q[0][2] = inputs.solution_values[q](2);
        Q[1][1] = inputs.solution_values[q](3);
        Q[1][2] = inputs.solution_values[q](4);
        Q[2][2] = -(Q[0][0] + Q[1][1]);

        eigs = dealii::eigenvectors(Q,
                                    dealii::SymmetricTensorEigenvectorMethod
                                    ::jacobi);

        // find index of maximal eigenvalue
        for (unsigned int i = 0; i < msc::mat_dim<dim>; ++i)
            if (eigs[i].first > eigs[max_eig_idx].first)
                max_eig_idx = i;

        computed_quantities[q](0) = eigs[max_eig_idx].second[0];
        computed_quantities[q](1) = eigs[max_eig_idx].second[1];
        if (dim == 3)
          computed_quantities[q](2) = eigs[max_eig_idx].second[2];

        computed_quantities[q](dim) = eigs[max_eig_idx].first;
    }
}

#endif
