#ifndef PLUS_HALF_ACTIVE_SOURCE_HPP
#define PLUS_HALF_ACTIVE_SOURCE_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <cmath>

template <int dim>
class PlusHalfActiveSource : public dealii::Function<dim>
{
public:

    PlusHalfActiveSource() : dealii::Function<dim>(dim + 1) {}

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;

    virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                            std::vector<double> &value_list,
                            const unsigned int component = 0) const override;

    virtual void vector_value(const dealii::Point<dim> &p,
                              dealii::Vector<double> &value) const override;

    virtual void vector_value_list(
        const std::vector<dealii::Point<dim>> &point_list,
        std::vector<dealii::Vector<double>> &value_list) const override;
};

template <int dim>
double
PlusHalfActiveSource<dim>::value(const dealii::Point<dim> &p /*p*/,
                          const unsigned int component /*component*/) const
{
    if (component == 0)
        return 1 / (2 * std::sqrt(p[0] * p[0] + p[1] * p[1]));
    else
        return 0;
}



template <int dim>
void PlusHalfActiveSource<dim>::
    value_list(const std::vector<dealii::Point<dim>> &point_list,
               std::vector<double> &value_list,
               const unsigned int component) const
{
    if (component == 0)
        for (unsigned int i = 0; i < point_list.size(); ++i)
            value_list[i] = 1 / (2 * std::sqrt(point_list[0] * point_list[0]
                                               + point_list[1] * point_list[1]));
    else
        for (unsigned int i = 0; i < point_list.size(); ++i)
            value_list[i] = 0;
}



template <int dim>
void PlusHalfActiveSource<dim>::vector_value(const dealii::Point<dim> &p,
                                             dealii::Vector<double> &values) const
{
    for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = PlusHalfActiveSource<dim>::value(p, c);
}



template <int dim>
void PlusHalfActiveSource<dim>::
    vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                      std::vector<dealii::Vector<double>> &value_list) const
{
    for (unsigned int i = 0; i < point_list.size(); ++i)
        PlusHalfActiveSource<dim>::vector_value(point_list[i], value_list[i]);
}

#endif
