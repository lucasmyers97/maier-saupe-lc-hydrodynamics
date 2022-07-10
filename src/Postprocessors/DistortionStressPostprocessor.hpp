#ifndef DISTORTION_STRESS_POSTPROCESSOR_HPP
#define DISTORTION_STRESS_POSTPROCESSOR_HPP

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
class DistortionStressPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    DistortionStressPostprocessor() {};

    virtual void evaluate_vector_field(
        const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<dealii::Vector<double>> &computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual dealii::UpdateFlags get_needed_update_flags() const override;
};



template <int dim>
dealii::UpdateFlags DistortionStressPostprocessor<dim>::get_needed_update_flags() const
{
    return dealii::update_gradients | dealii::update_quadrature_points;
}



template <int dim>
std::vector<std::string> DistortionStressPostprocessor<dim>::get_names() const
{
    std::vector<std::string> solution_names(msc::vec_dim<dim>,
                                            "elastic_stress_");
    for (std::size_t i = 0; i < solution_names.size(); ++i)
        solution_names[i] += std::to_string(i);

    return solution_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
DistortionStressPostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation
                ::DataComponentInterpretation>
        interpretation(msc::vec_dim<dim>,
                       dealii::DataComponentInterpretation
                       ::component_is_scalar);

    return interpretation;
}




template <int dim>
void DistortionStressPostprocessor<dim>::evaluate_vector_field
(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_gradients.size();
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
    Assert(inputs.solution_gradients[0].size() == msc::vec_dim<dim>,
           ExcInternalError());

    dealii::SymmetricTensor<2, dim> sigma_d;
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        sigma_d.clear();
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = i; j < dim; ++j)
            {
                for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
                    sigma_d[i][j] -= (2 * inputs.solution_gradients[q][k][i]
                                      * inputs.solution_gradients[q][k][j]);

                sigma_d[i][j] -= (inputs.solution_gradients[q][0][i]
                                  * inputs.solution_gradients[q][3][j]
                                  +
                                  inputs.solution_gradients[q][0][j]
                                  * inputs.solution_gradients[q][3][i]);
            }

        computed_quantities[q].reinit(msc::vec_dim<dim>);
        computed_quantities[q](0) = sigma_d[0][0];
        computed_quantities[q](1) = sigma_d[0][1];
        computed_quantities[q](3) = sigma_d[1][1];
        if (dim == 3)
        {
            computed_quantities[q](2) = sigma_d[0][2];
            computed_quantities[q](4) = sigma_d[1][2];
        }
        // for (unsigned int k = 0; k < msc::vec_dim<dim>; ++k)
        // {
        //   computed_quantities[q](0) -= (2 * inputs.solution_gradients[q][k][0]
        //                                 * inputs.solution_gradients[q][k][0]);
        //   computed_quantities[q](1) -= (2 * inputs.solution_gradients[q][k][0]
        //                                 * inputs.solution_gradients[q][k][1]);
        //   computed_quantities[q](3) -= (2 * inputs.solution_gradients[q][k][1]
        //                                 * inputs.solution_gradients[q][k][1]);
        //   if (dim == 3)
        //   {
        //       computed_quantities[q](2) -= (2 * inputs.solution_gradients[q][k][0]
        //                                     * inputs.solution_gradients[q][k][2]);
        //       computed_quantities[q](4) -= (2 * inputs.solution_gradients[q][k][1]
        //                                     * inputs.solution_gradients[q][k][2]);
        //   }
        // }
        // computed_quantities[q](0) -= (inputs.solution_gradients[q][0][0]
        //                               * inputs.solution_gradients[q][3][0]
        //                               +
        //                               inputs.solution_gradients[q][0][0]
        //                               * inputs.solution_gradients[q][3][0]);
        // computed_quantities[q](1) -= (inputs.solution_gradients[q][0][0]
        //                               * inputs.solution_gradients[q][3][1]
        //                               +
        //                               inputs.solution_gradients[q][0][1]
        //                               * inputs.solution_gradients[q][3][0]);
        // computed_quantities[q](3) -= (inputs.solution_gradients[q][0][1]
        //                               * inputs.solution_gradients[q][3][1]
        //                               +
        //                               inputs.solution_gradients[q][0][1]
        //                               * inputs.solution_gradients[q][3][1]);
        // if (dim == 3)
        // {
        //     computed_quantities[q](2) -= (inputs.solution_gradients[q][0][0] *
        //                                   inputs.solution_gradients[q][3][2] +
        //                                   inputs.solution_gradients[q][0][2] *
        //                                   inputs.solution_gradients[q][3][0]);
        //     computed_quantities[q](4) -= (inputs.solution_gradients[q][0][1] *
        //                                   inputs.solution_gradients[q][3][2] +
        //                                   inputs.solution_gradients[q][0][1] *
        //                                   inputs.solution_gradients[q][3][2]);
        // }
    }
}

#endif
