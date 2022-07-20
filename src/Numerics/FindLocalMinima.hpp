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
using cell_iterator = typename dealii::DoFHandler<dim>::cell_iterator;

template <int dim>
struct DefectQuantities
{
    double min_S;
    double max_D;
    dealii::Point<dim> min_pt;
};



/**
 * \brief iterates through each cell in triangulation attached to 
 * `dof_handler` and calculates S and D for that cell.
 * Values are stored in a vector of `DefectQuantities`, and each cell stores
 * its index via the `cell->set_user_index()` method.
 */
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

        cell->clear_user_index();
        cell->set_user_index(i);

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



template <int dim>
void clear_neighbor_flags(cell_iterator<dim> cell)
{
    if (!(cell->user_flag_set()))
        return;

    cell->clear_user_flag();
    for (unsigned int i = 0; i < cell->n_faces(); ++i)
    {
        if (cell->at_boundary(i))
            continue;

        auto neighbor = cell->neighbor(i);
        if (!neighbor->has_children())
            clear_neighbor_flags<dim>(neighbor);
        else
            for (unsigned int j = 0; j < neighbor->n_children(); ++j)
                clear_neighbor_flags<dim>(neighbor->child(j));
    }
}



/**
 * \brief Checks if `cell` contains the minimal value of S in a radius of
 * cells of size `R`.
 * It accomplishes this by recursively checking neighbors until it gets past
 * a distance `R`.
 */
template <int dim>
bool check_if_local_min(cell_iterator<dim> cell, double R, 
                        std::vector<DefectQuantities<dim>> &defect_quantities)
{
    unsigned int idx = cell->user_index();
    double S_val = defect_quantities[idx].min_S;
    dealii::Point<dim> pt = defect_quantities[idx].min_pt;

    cell->set_user_flag();

    bool is_local_min 
        =  check_pt_against_neighbors(cell, S_val, pt, R, defect_quantities);

    clear_neighbor_flags<dim>(cell);

    return is_local_min;
}



/**
 * \brief Checks possible minimum point defined by `S`, `pt`, and `R` against
 * the neighbors of `cell`.
 * Returns `true` if possible minimum is smaller than all neighbors of `cell`
 * within distance `R`, returns `false` if it finds an S-value smaller.
 */
template <int dim>
bool check_pt_against_neighbors(cell_iterator<dim> cell, 
                                double S, 
                                dealii::Point<dim> pt, 
                                double R,
                                std::vector<DefectQuantities<dim>> &defect_quantities)
{
    for (unsigned int i = 0; i < cell->n_faces(); ++i)
    {
        if (cell->at_boundary(i))
            continue;

        auto neighbor = cell->neighbor(i);
        if (!neighbor->has_children())
        {
            if (neighbor->user_flag_set())
                continue;
            if (!check_pt_against_cell(neighbor, S, pt, R, defect_quantities))
                return false;
        }
        else 
        {
            for (unsigned int j = 0; j < neighbor->n_children(); ++j)
            {
                auto child = neighbor->child(j);
                if (child->user_flag_set())
                    continue;
                if (!check_pt_against_cell(child, S, pt, R, defect_quantities))
                    return false;
            }
        }
    }

    return true;
}



/**
 * \brief Checks whether possible minimum defined by `S`, `pt`, and `R` is
 * smaller than all quadrature points in `cell`, and makes the same check
 * against neighbors of `cell`.
 */
template <int dim>
bool check_pt_against_cell(cell_iterator<dim> cell, 
                           double S, 
                           dealii::Point<dim> pt, 
                           double R, 
                           std::vector<DefectQuantities<dim>> &defect_quantities)
{
    // cell has been checked
    cell->set_user_flag();

    unsigned int idx = cell->user_index();
    double cell_S = defect_quantities[idx].min_S;
    dealii::Point<dim> cell_pt = defect_quantities[idx].min_pt;
    
    // don't care if cell is out of range
    if (cell_pt.distance(pt) > R)
        return true;
    if (cell_S < S)
        return false;
    if (!check_pt_against_neighbors(cell, S, pt, R, defect_quantities))
        return false;

    return true;
}




} // namespace NumericalTools

#endif
