#ifndef DISCLINATION_CHARGE_POSTPROCESSOR_HPP
#define DISCLINATION_CHARGE_POSTPROCESSOR_HPP

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <vector>
#include <string>
#include <array>
#include <utility>
#include <tuple>
#include <type_traits>

#include "Utilities/maier_saupe_constants.hpp"
#include "Numerics/DisclinationCharge.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
class DisclinationChargePostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    DisclinationChargePostprocessor() {};

    virtual void evaluate_vector_field(
        const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<dealii::Vector<double>> &computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual dealii::UpdateFlags get_needed_update_flags() const override;

private:
    double alpha;
};



template <int dim>
dealii::UpdateFlags DisclinationChargePostprocessor<dim>::get_needed_update_flags() const
{
    return (dealii::update_values
            | dealii::update_gradients
            | dealii::update_quadrature_points);
}



template <int dim>
std::vector<std::string> DisclinationChargePostprocessor<dim>::get_names() const
{
    std::vector<std::string> solution_names(msc::mat_dim<dim>*msc::mat_dim<dim>,
                                            "disclination_charge_");
    for (std::size_t i = 0; i < msc::mat_dim<dim>; ++i)
        for (std::size_t j = 0; j < msc::mat_dim<dim>; ++j)
            solution_names[i + msc::mat_dim<dim>*j] += std::to_string(i) + std::to_string(j);

    return solution_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
DisclinationChargePostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation
                ::DataComponentInterpretation>
        interpretation(msc::mat_dim<dim> * msc::mat_dim<dim>,
                       dealii::DataComponentInterpretation
                       ::component_is_scalar);

    return interpretation;
}




template <int dim>
void DisclinationChargePostprocessor<dim>::evaluate_vector_field
(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_gradients.size();
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
    Assert(inputs.solution_gradients[0].size() == msc::vec_dim<dim>,
           ExcInternalError());


    constexpr int D_dim = 3;
    dealii::Tensor<2, D_dim> D;

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        computed_quantities[q].reinit(msc::mat_dim<dim>*msc::mat_dim<dim>);
        NumericalTools::DisclinationCharge<dim>(inputs.solution_gradients[q], D);
        
        computed_quantities[q][0] = D[0][0];
        computed_quantities[q][1] = D[0][1];
        computed_quantities[q][2] = D[0][2];
        computed_quantities[q][3] = D[1][0];
        computed_quantities[q][4] = D[1][1];
        computed_quantities[q][5] = D[1][2];
        computed_quantities[q][6] = D[2][0];
        computed_quantities[q][7] = D[2][1];
        computed_quantities[q][8] = D[2][2];
    }
}

#endif
