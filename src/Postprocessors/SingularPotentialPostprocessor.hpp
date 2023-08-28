#ifndef SINGULAR_POTENTIAL_POSTPROCESSOR_HPP
#define SINGULAR_POTENTIAL_POSTPROCESSOR_HPP

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

#include "Numerics/LagrangeMultiplierAnalytic.hpp"
#include "Utilities/maier_saupe_constants.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
class SingularPotentialPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    SingularPotentialPostprocessor(LagrangeMultiplierAnalytic<dim>& singular_potential) 
        : singular_potential(singular_potential)
        {};

    inline virtual void evaluate_vector_field(
        const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<dealii::Vector<double>> &computed_quantities) const override;

    inline virtual std::vector<std::string> get_names() const override;

    inline virtual std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    inline virtual dealii::UpdateFlags get_needed_update_flags() const override;

private:
    LagrangeMultiplierAnalytic<dim>& singular_potential;
};



template <int dim>
dealii::UpdateFlags SingularPotentialPostprocessor<dim>::get_needed_update_flags() const
{
    return dealii::update_values | dealii::update_quadrature_points;
}



template <int dim>
std::vector<std::string> SingularPotentialPostprocessor<dim>::get_names() const
{
    std::vector<std::string> solution_names;
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        solution_names.emplace_back("Lambda" + std::to_string(i));

    return solution_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
SingularPotentialPostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interpretation(msc::vec_dim<dim>,
                   dealii::DataComponentInterpretation
                   ::component_is_scalar);

    return interpretation;
}




template <int dim>
inline void SingularPotentialPostprocessor<dim>::evaluate_vector_field
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

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {

        singular_potential.invertQ(inputs.solution_values[q]);
        singular_potential.returnLambda(computed_quantities[q]);
    }
}

#endif
