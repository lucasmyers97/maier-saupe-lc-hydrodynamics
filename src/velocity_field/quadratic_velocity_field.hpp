#ifndef QUADRATIC_VELOCITY_FIELD_HPP
#define QUADRATIC_VELOCITY_FIELD_HPP

#include "velocity_field.hpp"

#include <boost/any.hpp>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <string>
#include <map>

template <int dim>
class QuadraticVelocityField : public VelocityField<dim>
{
public:
    QuadraticVelocityField();
    QuadraticVelocityField(dealii::Tensor<1, dim> a,
                           dealii::Tensor<1, dim> b,
                           double max_flow_magnitude,
                           dealii::Tensor<1, dim> u);
    QuadraticVelocityField(std::map<std::string, boost::any> &am);

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;


    virtual void vector_value(const dealii::Point<dim> &p,
					          dealii::Vector<double> &value) const override;

    virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                            std::vector<double> &value_list,
                            const unsigned int component = 0) const override;

    virtual void
    vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                      std::vector<dealii::Vector<double>>   &value_list)
                      const override;

    /** Returns gradient of `component` evaluated at point `p` */
    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim> &p,
                                            const unsigned int component = 0) const override;

    /** Returns gradient of all components evaluated at point `p` */
    virtual void vector_gradient(const dealii::Point<dim> &p,
					           	 std::vector<dealii::Tensor<1, dim>> &gradients) const override;

    /** Set `gradients` to be the gradient of the `component` of the vector evaluated at each point in `point_list` */
    virtual void gradient_list(const std::vector<dealii::Point<dim>> &point_list,
                               std::vector<dealii::Tensor<1, dim>> &gradients,
                               const unsigned int component = 0) const override;

    /** For each component of the function, fill a vector of gradient values, one for each point. */
    virtual void vector_gradients(const std::vector<dealii::Point<dim>> &points,
					              std::vector<std::vector<dealii::Tensor<1, dim>>> &gradients) const override;

    /** Set gradients to the gradients of the function at the points, for all components. 
     *  It is assumed that gradients already has the right size, i.e. the same size as the points array.
     *
     *   The outer loop over gradients is over the points in the list, the inner loop over the different components of the function. 
     */
    virtual void
        vector_gradient_list(const std::vector<dealii::Point<dim>> &point_list,
                             std::vector<std::vector<dealii::Tensor<1, dim>>> &gradients) const override;

private:
    /** Representative no-slip endpoints. 
     *  The line connecting them should be perpendicular to the flow direction u.
     *  Magnitude of flow at any point is determined by projecting the point
     *  onto the axis connecting a and b, and then determining the distance
     *  from a (or b, equivalently) */
    dealii::Tensor<1, dim> a;
    dealii::Tensor<1, dim> b;

    /** length of the cavity, which is exactly |a - b| */
    double length;

    /** Magnitude of flow in the middle of the cavity */
    double max_flow_magnitude;

    /** Unit vector in the direction of flow */
    dealii::Tensor<1, dim> u;
};

#endif
