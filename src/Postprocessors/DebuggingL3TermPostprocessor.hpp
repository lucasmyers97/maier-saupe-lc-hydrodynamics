#ifndef DEBUGGING_L3_TERM_POSTPROCESSOR_HPP
#define DEBUGGING_L3_TERM_POSTPROCESSOR_HPP

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
class DebuggingL3TermPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    DebuggingL3TermPostprocessor() {};

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
dealii::UpdateFlags DebuggingL3TermPostprocessor<dim>::get_needed_update_flags() const
{
    return dealii::update_values 
        | dealii::update_quadrature_points
        | dealii::update_gradients;
}



template <int dim>
std::vector<std::string> DebuggingL3TermPostprocessor<dim>::get_names() const
{
    std::vector<std::string> solution_names;
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        solution_names.emplace_back("L3_old_" + std::to_string(i));
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        solution_names.emplace_back("L3_new_" + std::to_string(i));

    return solution_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
DebuggingL3TermPostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interpretation(2 * msc::vec_dim<dim>,
                   dealii::DataComponentInterpretation
                   ::component_is_scalar);

    return interpretation;
}




template <int dim>
inline void DebuggingL3TermPostprocessor<dim>::evaluate_vector_field
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

    const std::vector<std::vector<dealii::Tensor<1, dim> > >& dQ = inputs.solution_gradients;
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        computed_quantities[q][0] = (dQ[q][0][0]) * (dQ[q][0][0]) + dQ[q][0][1]*dQ[q][1][0] + (dQ[q][1][0]) * (dQ[q][1][0]) + dQ[q][1][1]*dQ[q][3][0] + (dQ[q][2][0]) * (dQ[q][2][0]) + dQ[q][2][1]*dQ[q][4][0];        
        computed_quantities[q][1] = dQ[q][0][0]*dQ[q][0][1] + dQ[q][0][1]*dQ[q][1][1] + dQ[q][1][0]*dQ[q][1][1] + dQ[q][1][1]*dQ[q][3][1] + dQ[q][2][0]*dQ[q][2][1] + dQ[q][2][1]*dQ[q][4][1]
                                    + dQ[q][0][0]*dQ[q][1][0] + dQ[q][1][0]*dQ[q][1][1] + dQ[q][1][0]*dQ[q][3][0] + dQ[q][2][0]*dQ[q][4][0] + dQ[q][3][0]*dQ[q][3][1] + dQ[q][4][0]*dQ[q][4][1];
        computed_quantities[q][2] = -(dQ[q][0][0] + dQ[q][3][0])*dQ[q][2][0] - (dQ[q][0][1] + dQ[q][3][1])*dQ[q][4][0] + dQ[q][0][0]*dQ[q][2][0] + dQ[q][1][0]*dQ[q][2][1] + dQ[q][1][0]*dQ[q][4][0] + dQ[q][3][0]*dQ[q][4][1];
        computed_quantities[q][3] = dQ[q][0][1]*dQ[q][1][0] + (dQ[q][1][1]) * (dQ[q][1][1]) + dQ[q][1][1]*dQ[q][3][0] + dQ[q][2][1]*dQ[q][4][0] + (dQ[q][3][1]) * (dQ[q][3][1]) + (dQ[q][4][1]) * (dQ[q][4][1]);
        computed_quantities[q][4] = -(dQ[q][0][0] + dQ[q][3][0])*dQ[q][2][1] - (dQ[q][0][1] + dQ[q][3][1])*dQ[q][4][1] + dQ[q][0][1]*dQ[q][2][0] + dQ[q][1][1]*dQ[q][2][1] + dQ[q][1][1]*dQ[q][4][0] + dQ[q][3][1]*dQ[q][4][1];

        computed_quantities[q][5] = (dQ[q][0][0] + dQ[q][3][0]) * (dQ[q][0][0] + dQ[q][3][0]) + (dQ[q][0][0]) * (dQ[q][0][0]) + 2*(dQ[q][1][0]) * (dQ[q][1][0]) + 2*(dQ[q][2][0]) * (dQ[q][2][0]) + (dQ[q][3][0]) * (dQ[q][3][0]) + 2*(dQ[q][4][0]) * (dQ[q][4][0]);
        computed_quantities[q][6] = (dQ[q][0][0] + dQ[q][3][0])*(dQ[q][0][1] + dQ[q][3][1]) + dQ[q][0][0]*dQ[q][0][1] + 2*dQ[q][1][0]*dQ[q][1][1] + 2*dQ[q][2][0]*dQ[q][2][1] + dQ[q][3][0]*dQ[q][3][1] + 2*dQ[q][4][0]*dQ[q][4][1];
        computed_quantities[q][7] = 0;
        computed_quantities[q][8] = (dQ[q][0][1] + dQ[q][3][1]) * (dQ[q][0][1] + dQ[q][3][1]) + (dQ[q][0][1]) * (dQ[q][0][1]) + 2*(dQ[q][1][1]) * (dQ[q][1][1]) + 2*(dQ[q][2][1]) * (dQ[q][2][1]) + (dQ[q][3][1]) * (dQ[q][3][1]) + 2*(dQ[q][4][1]) * (dQ[q][4][1]);
        computed_quantities[q][9] = 0;
    }
}

#endif
