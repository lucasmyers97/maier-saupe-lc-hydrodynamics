#include "MultiDefectConfiguration.hpp"

#include "Utilities/ParameterParser.hpp"

#include <deal.II/base/point.h>
#include <deal.II/base/geometric_utilities.h>

#include <stdexcept>
#include <vector>
#include <array>



template <int dim>
MultiDefectConfiguration<dim>::
MultiDefectConfiguration(const std::vector<dealii::Point<dim>> 
                         &defect_positions, 
                         const std::vector<double> &defect_charges,
                         const std::vector<double> &defect_orientations,
                         double S0,
                         double eps,
                         unsigned int degree,
                         unsigned int n_refines,
                         double tol,
                         unsigned int max_iter,
                         double newton_step)
    : BoundaryValues<dim>(std::string("dzyaloshinskii-function"))
    , defect_positions(defect_positions)
    , defect_charges(defect_charges)
    , defect_orientations(defect_orientations)
    , S0(S0)
    , eps(eps)
    , degree(degree)
    , n_refines(n_refines)
    , tol(tol)
    , max_iter(max_iter)
    , newton_step(newton_step)
{}



template <int dim>
MultiDefectConfiguration<dim>::
MultiDefectConfiguration(std::map<std::string, boost::any> &am)
    : BoundaryValues<dim>(std::string("multi-defect-configuration"),
                          boost::any_cast<std::string>(am["boundary-condition"]))
    , S0(boost::any_cast<double>(am["S-value"]))
    , defect_positions(ParameterParser::vector_to_dealii_point<dim>( 
                boost::any_cast<std::vector<std::vector<double>>>(am["defect-positions"])
                )
            )
    , defect_charges(boost::any_cast<std::vector<double>>(am["defect-charges"]))
    , defect_orientations(boost::any_cast<std::vector<double>>(am["defect-orientations"]))
    , defect_radius(boost::any_cast<double>(am["defect-radius"]))
    , eps(boost::any_cast<double>(am["anisotropy-eps"]))
    , degree(boost::any_cast<long>(am["degree"]))
    , n_refines(boost::any_cast<long>(am["n-refines"]))
    , tol(boost::any_cast<double>(am["tol"]))
    , max_iter(boost::any_cast<long>(am["max-iter"]))
    , newton_step(boost::any_cast<double>(am["newton-step"]))
{
    for (const auto &defect_position : defect_positions)
        this->defect_pts.push_back(defect_position);

    for (std::size_t i = 0; i < defect_positions.size(); ++i)
        for (std::size_t j = i + 1; j < defect_positions.size(); ++j)
            if (defect_positions[i].distance(defect_positions[j]) < defect_radius)
                throw std::logic_error("Initial defect positions too close");

    initialize();
}



template <int dim>
void MultiDefectConfiguration<dim>::
initialize()
{
    for (std::size_t i = 0; i < defect_positions.size(); ++i)
        dzyaloshinskii_systems.push_back(
                std::make_unique<DzyaloshinskiiSystem>(degree)
                );

    // run dzyaloshinskii sim
    for (std::size_t i = 0; i < dzyaloshinskii_systems.size(); ++i)
    {
        dzyaloshinskii_systems[i]->make_grid(n_refines);
        dzyaloshinskii_systems[i]->setup_system(defect_charges[i]);
        dzyaloshinskii_systems[i]->run_newton_method(eps, tol, max_iter, newton_step);

        // initialize fe_field_function
        dzyaloshinskii_functions.push_back( 
                std::make_unique<dealii::Functions::FEFieldFunction<1>>
                (dzyaloshinskii_systems[i]->return_dof_handler(), 
                 dzyaloshinskii_systems[i]->return_solution())
                );
    }
}



template <int dim>
inline double
MultiDefectConfiguration<dim>::
value_in_defect(const dealii::Functions::FEFieldFunction<1> &dzyaloshinskii_function,
                const dealii::Point<dim> &p,
                double defect_orientation,
                const unsigned int component) const
{
    const std::array<double, dim> p_sphere 
        = dealii::GeometricUtilities::Coordinates::to_spherical(p);

    dealii::Point<1> polar_angle( p_sphere[1] );
    double director_angle = dzyaloshinskii_function.value(polar_angle);
    director_angle += defect_orientation;

    double S = S0 * (2.0 / (1 + std::exp(-p_sphere[0])) - 1.0);

    switch (component)
    {
    case 0:
    	return 0.5 * S * ( 1.0/3.0 + std::cos(2*director_angle) );
    case 1:
    	return 0.5 * S * std::sin(2*director_angle);
    case 2:
    	return 0.0;
    case 3:
    	return 0.5 * S * ( 1.0/3.0 - std::cos(2*director_angle) );
    case 4:
    	return 0.0;
    }
}



template <int dim>
inline void
MultiDefectConfiguration<dim>::
vector_value_in_defect(const dealii::Functions::FEFieldFunction<1> 
                       &dzyaloshinskii_function,
                       const dealii::Point<dim> &p,
                       double defect_orientation,
                       dealii::Vector<double> &value) const
{
    const std::array<double, dim> p_sphere 
        = dealii::GeometricUtilities::Coordinates::to_spherical(p);

    dealii::Point<1> polar_angle( p_sphere[1] );
    double director_angle = dzyaloshinskii_function.value(polar_angle);
    director_angle += defect_orientation;

    double S = S0 * (2.0 / (1 + std::exp(-p_sphere[0])) - 1.0);

    value[0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*director_angle) );
    value[1] = 0.5 * S * std::sin(2*director_angle);
    value[2] = 0.0;
    value[3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*director_angle) );
    value[4] = 0.0;
}



template <int dim>
inline double
MultiDefectConfiguration<dim>::
value_outside_defect(const dealii::Point<dim> &p, const unsigned int component) const
{
    double return_value = 0.0;
    for (std::size_t i = 0; i < defect_positions.size(); ++i)
    {
        const std::array<double, dim> p_sphere 
            = dealii::GeometricUtilities::Coordinates::
              to_spherical( dealii::Point<dim>(p - defect_positions[i]) );
        dealii::Point<1> polar_angle(p_sphere[1]);

        double director_angle = dzyaloshinskii_functions[i]->value(polar_angle);
        director_angle += defect_orientations[i];

        double S_at_defect_radius = S0 * (2.0 / (1 + std::exp(-defect_radius)) - 1.0);
        double S = S_at_defect_radius * std::exp(-(p_sphere[0] - defect_radius));

        switch (component)
        {
        case 0:
        	return_value += 0.5 * S * ( 1.0/3.0 + std::cos(2*director_angle) );
            break;
        case 1:
        	return_value += 0.5 * S * std::sin(2*director_angle);
            break;
        case 2:
        	return_value += 0.0;
            break;
        case 3:
        	return_value += 0.5 * S * ( 1.0/3.0 - std::cos(2*director_angle) );
            break;
        case 4:
        	return_value += 0.0;
            break;
        }
    }

    return return_value;
}



template <int dim>
inline void
MultiDefectConfiguration<dim>::
vector_value_outside_defect(const dealii::Point<dim> &p, 
                            dealii::Vector<double> &value) const
{
    value = 0;
    for (std::size_t i = 0; i < defect_positions.size(); ++i)
    {
        const std::array<double, dim> p_sphere 
            = dealii::GeometricUtilities::Coordinates::
              to_spherical( dealii::Point<dim>(p - defect_positions[i]) );
        dealii::Point<1> polar_angle(p_sphere[1]);

        double director_angle = dzyaloshinskii_functions[i]->value(polar_angle);
        director_angle += defect_orientations[i];

        double S_at_defect_radius = S0 * (2.0 / (1 + std::exp(-defect_radius)) - 1.0);
        double S = S_at_defect_radius * std::exp(-(p_sphere[0] - defect_radius));

        value[0] += 0.5 * S * ( 1.0/3.0 + std::cos(2*director_angle) );
        value[1] += 0.5 * S * std::sin(2*director_angle);
        value[2] += 0.0;
        value[3] += 0.5 * S * ( 1.0/3.0 - std::cos(2*director_angle) );
        value[4] += 0.0;
    }
}



template <int dim>
double MultiDefectConfiguration<dim>::
value(const dealii::Point<dim> &p,
      const unsigned int component) const
{
    for (std::size_t i = 0; i < defect_positions.size(); ++i)
    {
        if (defect_positions[i].distance(p) > defect_radius)
            continue;

        return value_in_defect(*dzyaloshinskii_functions[i], 
                               dealii::Point<dim>(p - defect_positions[i]),
                               defect_orientations[i],
                               component);
    }

    return value_outside_defect(p, component);
};



template <int dim>
void MultiDefectConfiguration<dim>::
vector_value(const dealii::Point<dim> &p, dealii::Vector<double> &value) const
{
    if (value.size() != 5)
        throw std::invalid_argument("Input to MultiDefectConfiguration "
                                    "vector_value has incorrect length.");

    for (std::size_t i = 0; i < defect_positions.size(); ++i)
    {
        if (defect_positions[i].distance(p) > defect_radius)
            continue;

        vector_value_in_defect(*dzyaloshinskii_functions[i], 
                               dealii::Point<dim>(p - defect_positions[i]),
                               defect_orientations[i],
                               value);
        return;
    }

    vector_value_outside_defect(p, value);
}



template <int dim>
void MultiDefectConfiguration<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list, 
           std::vector<double> &value_list,
           const unsigned int component) const
{
    if (point_list.size() != value_list.size())
        throw std::invalid_argument("point_list and value_list have different "
                                    "lengths in MultiDefectConfiguration.");

    for (std::size_t pt_idx = 0; pt_idx < point_list.size(); ++pt_idx)
        value_list[pt_idx] = value(point_list[pt_idx], component);
}



template <int dim>
void MultiDefectConfiguration<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &points, 
                  std::vector<dealii::Vector<double>> &values) const
{
    if (points.size() != values.size())
        throw std::invalid_argument("points and values have different lengths "
                                    "in MultiDefectConfiguration.");

    for (std::size_t pt_idx = 0; pt_idx < points.size(); ++pt_idx)
        vector_value(points[pt_idx], values[pt_idx]);
}

template class MultiDefectConfiguration<3>;
template class MultiDefectConfiguration<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(MultiDefectConfiguration<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(MultiDefectConfiguration<3>)
