#ifndef STEP_55_RIGHT_HAND_SIDE_HPP
#define STEP_55_RIGHT_HAND_SIDE_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/numbers.h>

#include <cmath>

template <int dim>
class Step55RightHandSide : public dealii::Function<dim>
{
public:
    Step55RightHandSide()
        : dealii::Function<dim>(dim + 1)
    {}

    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &  value) const override;
};



template <int dim>
void Step55RightHandSide<dim>::
vector_value(const dealii::Point<dim> &p,
             dealii::Vector<double> &  values) const
{
    const double R_x = p[0];
    const double R_y = p[1];

    constexpr double pi  = dealii::numbers::PI;
    constexpr double pi2 = dealii::numbers::PI * dealii::numbers::PI;

  values[0] = -1.0L / 2.0L * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0) *
                std::exp(R_x * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0)) -
              0.4 * pi2 * std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                std::cos(2 * R_y * pi) +
              0.1 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 2) *
                std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                std::cos(2 * R_y * pi);
  values[1] = 0.2 * pi * (-std::sqrt(25.0 + 4 * pi2) + 5.0) *
                std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                std::sin(2 * R_y * pi) -
              0.05 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 3) *
                std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                std::sin(2 * R_y * pi) / pi;
  values[2] = 0;
}

#endif
