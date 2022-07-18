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
    std::vector<std::string> solution_names(msc::vec_dim<dim> + 1,
                                            "disclination_charge_");
    for (std::size_t i = 0; i < solution_names.size(); ++i)
        solution_names[i] += std::to_string(i);

    return solution_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
DisclinationChargePostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation
                ::DataComponentInterpretation>
        interpretation(msc::vec_dim<dim> + 1,
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
    dealii::Tensor<3, D_dim> dQ;
    dealii::Tensor<2, D_dim> D;

    dealii::Tensor<3, D_dim> eps;
    std::vector<std::tuple<int, int, int>> 
        pos_perms = {{0, 1, 2},
                     {1, 2, 0},
                     {2, 0, 1}};
    std::vector<std::tuple<int, int, int>> 
        neg_perms = {{2, 1, 0},
                     {0, 2, 1},
                     {1, 0, 2}};
    std::tuple<int, int, int> perm;
    for (int i = 0; i < D_dim; ++i)
        for (int j = 0; j < D_dim; ++j)
            for (int k = 0; k < D_dim; ++k)
            {
                perm = {i, j, k};
                if ((perm == pos_perms[0]) ||
                    (perm == pos_perms[1]) ||
                    (perm == pos_perms[2]))

                    eps[i][j][k] = 1;

                else if ((perm == neg_perms[0]) ||
                         (perm == neg_perms[1]) ||
                         (perm == neg_perms[2]))

                    eps[i][j][k] = -1;
            }

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        for (unsigned int k = 0; k < dim; ++k)
        {
            dQ[k][0][0] = inputs.solution_gradients[q][0][k];
            dQ[k][0][1] = inputs.solution_gradients[q][1][k];
            dQ[k][1][0] = dQ[k][0][1];
            dQ[k][1][1] = inputs.solution_gradients[q][3][k];

            if (dim == 3)
            {
                dQ[k][0][2] = inputs.solution_gradients[q][2][k];
                dQ[k][2][0] = dQ[k][0][2];
                dQ[k][1][2] = inputs.solution_gradients[q][4][k];
                dQ[k][2][1] = dQ[k][1][2];
                dQ[k][2][2] = -(dQ[k][0][0] + dQ[k][1][1]);
            }
        }

        computed_quantities[q].reinit(msc::vec_dim<dim> + 1);
        NumericalTools::DisclinationCharge<dim>(inputs.solution_gradients[q], D);
        
        computed_quantities[q][0] = D[0][0];
        computed_quantities[q][1] = D[0][1];
        computed_quantities[q][2] = D[0][2];
        computed_quantities[q][3] = D[1][1];
        computed_quantities[q][4] = D[1][2];
        computed_quantities[q][5] = D[2][2];
    }
}

#endif
