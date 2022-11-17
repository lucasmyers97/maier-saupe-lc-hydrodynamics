#ifndef ENERGY_POSTPROCESSOR_HPP
#define ENERGY_POSTPROCESSOR_HPP

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
#include <cmath>

#include "Utilities/maier_saupe_constants.hpp"
#include "Numerics/LagrangeMultiplierAnalytic.hpp"

namespace msc = maier_saupe_constants;

template <int dim>
class EnergyPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    EnergyPostprocessor(LagrangeMultiplierAnalytic<dim> lma_, 
                        double alpha_,
                        double L2_,
                        double L3_) 
        : lma(lma_)
        , alpha(alpha_)
        , L2(L2_)
        , L3(L3_)
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
    static const unsigned int n_energy_components = 5;

    mutable LagrangeMultiplierAnalytic<dim> lma;
    const double alpha;
    const double L2;
    const double L3;
};



template <int dim>
dealii::UpdateFlags EnergyPostprocessor<dim>::get_needed_update_flags() const
{
    return dealii::update_values | dealii::update_quadrature_points;
}



template <int dim>
std::vector<std::string> EnergyPostprocessor<dim>::get_names() const
{
    std::vector<std::string> energy_names;
    energy_names.emplace_back("mean_field_term");
    energy_names.emplace_back("entropy_term");
    energy_names.emplace_back("L1_elastic_term");
    energy_names.emplace_back("L2_elastic_term");
    energy_names.emplace_back("L3_elastic_term");

    return energy_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
EnergyPostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interpretation(n_energy_components,
                   dealii::DataComponentInterpretation
                   ::component_is_scalar);

    return interpretation;
}




template <int dim>
inline void EnergyPostprocessor<dim>::evaluate_vector_field
(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_values.size();
    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == n_energy_components,
           ExcInternalError());

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    dealii::Vector<double> Lambda_vec(msc::vec_dim<dim>);
    double Z;
    std::vector<dealii::Tensor<1, dim>> dQ(msc::vec_dim<dim>);
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        Q_vec = inputs.solution_values[q];
        dQ = inputs.solution_gradients[q];

        Lambda_vec = 0;
        Z = 0;
        lma.invertQ(Q_vec);
        lma.returnLambda(Lambda_vec);
        Z = lma.returnZ();

        /* mean field term */
        computed_quantities[q](0) = 
                (alpha*(-(Q_vec[0]) * (Q_vec[0]) 
                        - Q_vec[0]*Q_vec[3] 
                        - (Q_vec[1]) * (Q_vec[1]) 
                        - (Q_vec[2]) * (Q_vec[2]) 
                        - (Q_vec[3]) * (Q_vec[3]) 
                        - (Q_vec[4]) * (Q_vec[4])));

        /* entropy term */
        computed_quantities[q](1) =
                (2*Q_vec[0]*Lambda_vec[0] + Q_vec[0]*Lambda_vec[3] 
                 + 2*Q_vec[1]*Lambda_vec[1] + 2*Q_vec[2]*Lambda_vec[2] 
                 + Q_vec[3]*Lambda_vec[0] + 2*Q_vec[3]*Lambda_vec[3] 
                 + 2*Q_vec[4]*Lambda_vec[4] - std::log(Z) + std::log(4*M_PI));

        /* L1 elastic term */
        computed_quantities[q](2) =
                ((1.0/2.0)*(-dQ[0][0] - dQ[3][0]) 
                 * (-dQ[0][0] - dQ[3][0]) 
                 + (1.0/2.0)*(-dQ[0][1] - dQ[3][1]) 
                 * (-dQ[0][1] - dQ[3][1]) 
                 + (1.0/2.0)*(dQ[0][0]) 
                 * (dQ[0][0]) 
                 + (1.0/2.0)*(dQ[0][1]) * (dQ[0][1]) 
                 + (dQ[1][0]) * (dQ[1][0]) 
                 + (dQ[1][1]) * (dQ[1][1]) 
                 + (dQ[2][0]) * (dQ[2][0]) 
                 + (dQ[2][1]) * (dQ[2][1]) 
                 + (1.0/2.0)*(dQ[3][0]) * (dQ[3][0]) 
                 + (1.0/2.0)*(dQ[3][1]) * (dQ[3][1]) 
                 + (dQ[4][0]) * (dQ[4][0]) 
                 + (dQ[4][1]) * (dQ[4][1]));

        /* L2 elastic term */
        computed_quantities[q](3) =
                ((1.0/2.0)*L2
                 * ((dQ[0][0] + dQ[1][1]) * (dQ[0][0] + dQ[1][1]) 
                 + (dQ[1][0] + dQ[3][1]) * (dQ[1][0] + dQ[3][1]) 
                 + (dQ[2][0] + dQ[4][1]) * (dQ[2][0] + dQ[4][1])));

        /* L3 elastic term */
        computed_quantities[q](4) = 
                ((1.0/2.0)*L3
                 *(2*((-dQ[0][0] - dQ[3][0])*(-dQ[0][1] - dQ[3][1]) 
                         + dQ[0][0]*dQ[0][1] + 2*dQ[1][0]*dQ[1][1] 
                         + 2*dQ[2][0]*dQ[2][1] + dQ[3][0]*dQ[3][1] 
                         + 2*dQ[4][0]*dQ[4][1])*Q_vec[1] 
                     + ((-dQ[0][0] - dQ[3][0]) 
                         * (-dQ[0][0] - dQ[3][0]) 
                         + (dQ[0][0]) * (dQ[0][0]) 
                         + 2*(dQ[1][0]) * (dQ[1][0]) 
                         + 2*(dQ[2][0]) * (dQ[2][0]) 
                         + (dQ[3][0]) * (dQ[3][0]) 
                         + 2*(dQ[4][0]) * (dQ[4][0]))*Q_vec[0] 
                     + ((-dQ[0][1] - dQ[3][1]) 
                         * (-dQ[0][1] - dQ[3][1]) 
                         + (dQ[0][1]) * (dQ[0][1]) 
                         + 2*(dQ[1][1]) * (dQ[1][1]) 
                         + 2*(dQ[2][1]) * (dQ[2][1]) 
                         + (dQ[3][1]) * (dQ[3][1]) 
                         + 2*(dQ[4][1]) * (dQ[4][1]))*Q_vec[3]));
    }
}

#endif
