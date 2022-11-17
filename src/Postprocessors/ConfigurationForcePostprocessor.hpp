#ifndef CONFIGURATION_FORCE_POSTPROCESSOR_HPP
#define CONFIGURATION_FORCE_POSTPROCESSOR_HPP

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
class ConfigurationForcePostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    ConfigurationForcePostprocessor(LagrangeMultiplierAnalytic<dim> lma_, 
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
dealii::UpdateFlags ConfigurationForcePostprocessor<dim>::get_needed_update_flags() const
{
    return dealii::update_values
           | dealii::update_gradients
           | dealii::update_hessians
           | dealii::update_quadrature_points;
}



template <int dim>
std::vector<std::string> ConfigurationForcePostprocessor<dim>::get_names() const
{
    std::vector<std::string> configuration_force_names;
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        configuration_force_names.emplace_back("mean_field_term_" 
                                               + std::to_string(i));
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        configuration_force_names.emplace_back("entropy_term_" 
                                               + std::to_string(i));
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        configuration_force_names.emplace_back("L1_elastic_term_" 
                                               + std::to_string(i));
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        configuration_force_names.emplace_back("L2_elastic_term_" 
                                               + std::to_string(i));
    for (unsigned int i = 0; i < msc::vec_dim<dim>; ++i)
        configuration_force_names.emplace_back("L3_elastic_term_" 
                                               + std::to_string(i));

    return configuration_force_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
ConfigurationForcePostprocessor<dim>::get_data_component_interpretation()
  const
{
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    interpretation(n_energy_components * msc::vec_dim<dim>,
                   dealii::DataComponentInterpretation
                   ::component_is_scalar);

    return interpretation;
}




template <int dim>
inline void ConfigurationForcePostprocessor<dim>::evaluate_vector_field
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

    dealii::Vector<double> Q_vec(msc::vec_dim<dim>);
    dealii::Vector<double> Lambda_vec(msc::vec_dim<dim>);
    std::vector<dealii::Tensor<1, dim>> dQ(msc::vec_dim<dim>);
    std::vector<dealii::Tensor<2, dim>> ddQ(msc::vec_dim<dim>);

    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        Q_vec = inputs.solution_values[q];
        dQ = inputs.solution_gradients[q];
        ddQ = inputs.solution_hessians[q];

        Lambda_vec = 0;
        lma.invertQ(Q_vec);
        lma.returnLambda(Lambda_vec);

        /* mean field term */
        computed_quantities[q](0) =
            (alpha*Q_vec[0]);
        computed_quantities[q](1) =
            (alpha*Q_vec[1]);
        computed_quantities[q](2) =
            (alpha*Q_vec[2]);
        computed_quantities[q](3) =
            (alpha*Q_vec[3]);
        computed_quantities[q](4) =
            (alpha*Q_vec[4]);
        
        /* entropy term */
        computed_quantities[q](5) =
            (-Lambda_vec[0]);
        computed_quantities[q](6) =
            (-Lambda_vec[1]);
        computed_quantities[q](7) =
            (-Lambda_vec[2]);
        computed_quantities[q](8) =
            (-Lambda_vec[3]);
        computed_quantities[q](9) =
            (-Lambda_vec[4]);
        
        /* L1 elastic term */
        computed_quantities[q](10) =
            (ddQ[0][0][0] + ddQ[0][1][1]);
        computed_quantities[q](11) =
            (ddQ[1][0][0] + ddQ[1][1][1]);
        computed_quantities[q](12) =
            (ddQ[2][0][0] + ddQ[2][1][1]);
        computed_quantities[q](13) =
            (ddQ[3][0][0] + ddQ[3][1][1]);
        computed_quantities[q](14) =
            (ddQ[4][0][0] + ddQ[4][1][1]);
        
        /* L2 elastic term */
        computed_quantities[q](15) =
            ((1.0/3.0)*L2*(2*ddQ[0][0][0] - ddQ[3][1][1] + ddQ[1][0][1]));
        computed_quantities[q](16) =
            ((1.0/2.0)*L2*(ddQ[1][0][0] + ddQ[1][1][1] + ddQ[0][0][1] + ddQ[3][0][1]));
        computed_quantities[q](17) =
            ((1.0/2.0)*L2*(ddQ[2][0][0] + ddQ[4][0][1]));
        computed_quantities[q](18) =
            ((1.0/3.0)*L2*(-ddQ[0][0][0] + 2*ddQ[3][1][1] + ddQ[1][0][1]));
        computed_quantities[q](19) =
            ((1.0/2.0)*L2*(ddQ[4][1][1] + ddQ[2][0][1]));
        
        /* L3 elastic term */
        computed_quantities[q](20) =
            ((1.0/3.0)*L3*(3*Q_vec[0]*ddQ[0][0][0] + 6*Q_vec[1]*ddQ[0][0][1] + 3*Q_vec[3]*ddQ[0][1][1] + (dQ[0][0]) * (dQ[0][0]) + 3*dQ[0][0]*dQ[1][1] - 2*dQ[0][0]*dQ[3][0] + (dQ[0][1]) * (dQ[0][1]) + 3*dQ[0][1]*dQ[1][0] + 4*dQ[0][1]*dQ[3][1] - 2*(dQ[1][0]) * (dQ[1][0]) + (dQ[1][1]) * (dQ[1][1]) - 2*(dQ[2][0]) * (dQ[2][0]) + (dQ[2][1]) * (dQ[2][1]) - 2*(dQ[3][0]) * (dQ[3][0]) + (dQ[3][1]) * (dQ[3][1]) - 2*(dQ[4][0]) * (dQ[4][0]) + (dQ[4][1]) * (dQ[4][1])));
        computed_quantities[q](21) =
            ((1.0/2.0)*L3*(-(dQ[0][0] + dQ[3][0])*(dQ[0][1] + dQ[3][1]) + 2*Q_vec[0]*ddQ[1][0][0] + 4*Q_vec[1]*ddQ[1][0][1] + 2*Q_vec[3]*ddQ[1][1][1] - dQ[0][0]*dQ[0][1] + 2*dQ[0][0]*dQ[1][0] + 2*dQ[1][0]*dQ[1][1] + 2*dQ[1][1]*dQ[3][1] - 2*dQ[2][0]*dQ[2][1] - dQ[3][0]*dQ[3][1] - 2*dQ[4][0]*dQ[4][1]));
        computed_quantities[q](22) =
            (L3*(Q_vec[0]*ddQ[2][0][0] + 2*Q_vec[1]*ddQ[2][0][1] + Q_vec[3]*ddQ[2][1][1] + dQ[0][0]*dQ[2][0] + dQ[1][0]*dQ[2][1] + dQ[1][1]*dQ[2][0] + dQ[2][1]*dQ[3][1]));
        computed_quantities[q](23) =
            ((1.0/3.0)*L3*(3*Q_vec[0]*ddQ[3][0][0] + 6*Q_vec[1]*ddQ[3][0][1] + 3*Q_vec[3]*ddQ[3][1][1] + (dQ[0][0]) * (dQ[0][0]) + 4*dQ[0][0]*dQ[3][0] - 2*(dQ[0][1]) * (dQ[0][1]) - 2*dQ[0][1]*dQ[3][1] + (dQ[1][0]) * (dQ[1][0]) + 3*dQ[1][0]*dQ[3][1] - 2*(dQ[1][1]) * (dQ[1][1]) + 3*dQ[1][1]*dQ[3][0] + (dQ[2][0]) * (dQ[2][0]) - 2*(dQ[2][1]) * (dQ[2][1]) + (dQ[3][0]) * (dQ[3][0]) + (dQ[3][1]) * (dQ[3][1]) + (dQ[4][0]) * (dQ[4][0]) - 2*(dQ[4][1]) * (dQ[4][1])));
        computed_quantities[q](24) =
            (L3*(Q_vec[0]*ddQ[4][0][0] + 2*Q_vec[1]*ddQ[4][0][1] + Q_vec[3]*ddQ[4][1][1] + dQ[0][0]*dQ[4][0] + dQ[1][0]*dQ[4][1] + dQ[1][1]*dQ[4][0] + dQ[3][1]*dQ[4][1]));}
}

#endif
