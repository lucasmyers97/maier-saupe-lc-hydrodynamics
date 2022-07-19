#ifndef FIND_LOCAL_MINIMA_HPP
#define FIND_LOCAL_MINIMA_HPP

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/dofs/dof_handler.h>
#include <vector>

#include "Numerics/CalcSValue.hpp"
#include "Numerics/DisclinationCharge.hpp"
#include "Utilities/maier_saupe_constants.hpp"

namespace NumericalTools
{
    namespace msc = maier_saupe_constants;

    template <int dim>
    struct DefectQuantities
    {
        double min_S;
        double max_D;
        dealii::Point<dim> min_pt;
    };

    template <int dim>
    std::vector<DefectQuantities<dim>> 
    calculate_defect_quantities(const dealii::DoFHandler<dim> &dof_handler,
                                const dealii::TrilinosWrappers::MPI::Vector &solution)
    {
        dealii::QGauss<dim> quadrature_formula(dof_handler.get_fe().degree + 2);
        dealii::FEValues<dim> fe_values(dof_handler.get_fe(),
                                        quadrature_formula,
                                        dealii::update_quadrature_points |
                                        dealii::update_values |
                                        dealii::update_gradients);

        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_Q_components 
            = dof_handler.get_fe().n_components();

        std::vector<dealii::Vector<double>> 
            Q_vec(n_q_points, dealii::Vector<double>(n_Q_components));
        std::vector<std::vector<dealii::Tensor<1, dim>>> 
            dQ(n_q_points, std::vector<dealii::Tensor<1, dim>>(n_Q_components));

        dealii::SymmetricTensor<2, msc::mat_dim<dim>> Q;
        std::vector<double> S(n_q_points);
        dealii::Tensor<2, msc::mat_dim<dim>, double> D;
        std::vector<double> charge_values(n_q_points);

        std::vector<DefectQuantities<dim>> 
            defect_quantities(dof_handler.get_triangulation().n_active_cells());

        dealii::types::global_cell_index i = 0;
        for (auto cell = dof_handler.begin_active(); 
             cell != dof_handler.end(); ++cell, ++i)
        {
            if (!cell->is_locally_owned())
                continue;

            fe_values.reinit(cell);
            fe_values.get_function_values(solution, Q_vec);
            fe_values.get_function_gradients(solution, dQ);

            for (std::size_t q = 0; q < n_q_points; ++q)
            {
                Q[0][0] = Q_vec[q][0];
                Q[0][1] = Q_vec[q][1];
                Q[1][1] = Q_vec[q][3];
                Q[0][2] = Q_vec[q][2];
                Q[1][2] = Q_vec[q][4];
                Q[2][2] = -(Q[0][0] + Q[1][1]);

                S[q] = calcSValue<dim>(Q);
                DisclinationCharge(dQ[q], D);
                charge_values[q] = D[2][2];
            }
            auto min_S_iter = std::min_element(S.begin(), S.end());
            auto max_D_iter = std::max_element(charge_values.begin(),
                                               charge_values.end());
            auto min_S_idx = std::distance(S.begin(), min_S_iter);
            auto max_D_idx = std::distance(charge_values.begin(), max_D_iter);

            defect_quantities[i].min_S = S[min_S_idx];
            defect_quantities[i].max_D = charge_values[max_D_idx];
            defect_quantities[i].min_pt = fe_values.quadrature_point(min_S_idx);
        }
        
        return defect_quantities;
    } // calculate_defect_quantities

} // namespace NumericalTools

#endif
