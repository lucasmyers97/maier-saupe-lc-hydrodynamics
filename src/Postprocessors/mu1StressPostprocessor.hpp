#ifndef MU1_STRESS_POSTPROCESSOR_HPP
#define MU1_STRESS_POSTPROCESSOR_HPP

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

namespace msc = maier_saupe_constants;

template <int dim>
class mu1StressPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    mu1StressPostprocessor(const LagrangeMultiplierAnalytic<dim> lma_,
                           const double alpha_)
        : lma(lma_)
        , alpha(alpha_)
    {};

    virtual void evaluate_vector_field(
        const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<dealii::Vector<double>> &computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual dealii::UpdateFlags get_needed_update_flags() const override;

private:
    mutable LagrangeMultiplierAnalytic<dim> lma;
    double alpha;
};



template <int dim>
dealii::UpdateFlags mu1StressPostprocessor<dim>::get_needed_update_flags() const
{
    return (dealii::update_values
            | dealii::update_hessians
            | dealii::update_quadrature_points);
}



template <int dim>
std::vector<std::string> mu1StressPostprocessor<dim>::get_names() const
{
    std::vector<std::string> solution_names(msc::vec_dim<dim>,
                                            "mu1_stress_");
    for (std::size_t i = 0; i < solution_names.size(); ++i)
        solution_names[i] += std::to_string(i);

    return solution_names;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
mu1StressPostprocessor<dim>::get_data_component_interpretation()
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
void mu1StressPostprocessor<dim>::evaluate_vector_field
(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<dealii::Vector<double>> &computed_quantities) const
{
    const unsigned int n_quadrature_points = inputs.solution_gradients.size();
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
    Assert(inputs.solution_gradients[0].size() == msc::vec_dim<dim>,
           ExcInternalError());

    dealii::SymmetricTensor<2, dim> H;
    dealii::Vector<double> Q(msc::vec_dim<dim>);
    dealii::Vector<double> Lambda(msc::vec_dim<dim>);
    dealii::Vector<double> lap_Q(msc::vec_dim<dim>);
    dealii::Tensor<2, dim> Q_mat;
    dealii::Tensor<2, dim> mu2_tensor;
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
        Q = inputs.solution_values[q];
        lma.invertQ(Q);
        lma.returnLambda(Lambda);

        lap_Q = 0;
        for (std::size_t k = 0; k < lap_Q.size(); ++k)
            for (int i = 0; i < dim; ++i)
                lap_Q[k] += inputs.solution_hessians[q][k][i][i];

        H.clear();
        H[0][0] = (alpha * Q[0]
                   + lap_Q[0] - Lambda[0]);
        H[0][1] = (alpha * Q[1]
                   + lap_Q[1] - Lambda[1]);
        H[1][1] = (alpha * Q[3]
                   + lap_Q[3] - Lambda[3]);
        if (dim == 3)
        {
            H[0][2] = (alpha * Q[2]
                       + lap_Q[2] - Lambda[2]);
            H[1][2] = (alpha * Q[4]
                       + lap_Q[4] - Lambda[4]);
            H[2][2] = -(H[0][0] + H[1][1]);
        }

        Q_mat.clear();
        Q_mat[0][0] = Q[0];
        Q_mat[0][1] = Q[1];
        Q_mat[1][0] = Q_mat[0][1];
        Q_mat[1][1] = Q[3];
        if (dim == 3)
        {
            Q_mat[0][2] = Q[2];
            Q_mat[1][2] = Q[4];
            Q_mat[2][0] = Q_mat[0][2];
            Q_mat[2][1] = Q_mat[1][2];
            Q_mat[2][2] = -(Q_mat[0][0] + Q_mat[1][1]);
        }

        mu2_tensor = H*Q_mat - Q_mat*H;

        computed_quantities[q].reinit(msc::vec_dim<dim>);
        computed_quantities[q](0) = mu2_tensor[0][0];
        computed_quantities[q](1) = mu2_tensor[0][1];
        computed_quantities[q](3) = mu2_tensor[1][1];
        if (dim == 3)
        {
            computed_quantities[q](2) = mu2_tensor[0][2];
            computed_quantities[q](4) = mu2_tensor[1][2];
        }
    }
}

#endif
