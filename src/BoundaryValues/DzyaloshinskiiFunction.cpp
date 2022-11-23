#include "DzyaloshinskiiFunction.hpp"

#include <deal.II/base/point.h>



template <int dim>
DzyaloshinskiiFunction<dim>::
DzyaloshinskiiFunction(const dealii::Point<dim> &p, 
                       double S0_,
                       double eps_,
                       unsigned int degree_,
                       double charge_,
                       unsigned int n_refines_,
                       double tol_,
                       unsigned int max_iter_,
                       double newton_step_)
    : BoundaryValues<dim>(std::string("dzyaloshinskii-function"))
    , defect_center(p)
    , S0(S0_)
    , eps(eps_)
    , degree(degree_)
    , charge(charge_)
    , n_refines(n_refines_)
    , tol(tol_)
    , max_iter(max_iter_)
    , newton_step(newton_step_)
{}



template <int dim>
DzyaloshinskiiFunction<dim>::
DzyaloshinskiiFunction(std::map<std::string, boost::any> &am)
    : BoundaryValues<dim>(std::string("dzyaloshinskii-function"),
                          boost::any_cast<std::string>(am["boundary-condition"]))
    , S0(boost::any_cast<double>(am["S-value"]))
    , defect_center(boost::any_cast<double>(am["x"]),
                    boost::any_cast<double>(am["y"]))
    , eps(boost::any_cast<double>(am["anisotropy-eps"]))
    , degree(boost::any_cast<long>(am["degree"]))
    , charge(boost::any_cast<double>(am["charge"]))
    , n_refines(boost::any_cast<long>(am["n-refines"]))
    , tol(boost::any_cast<double>(am["tol"]))
    , max_iter(boost::any_cast<long>(am["max-iter"]))
    , newton_step(boost::any_cast<double>(am["newton-step"]))
{
    initialize();
}



template <int dim>
void DzyaloshinskiiFunction<dim>::
initialize()
{
    dzyaloshinskii_system
        = std::make_unique<DzyaloshinskiiSystem>(degree);

    // run dzyaloshinskii sim
    dzyaloshinskii_system->make_grid(n_refines);
    dzyaloshinskii_system->setup_system(charge);
    dzyaloshinskii_system->run_newton_method(eps, tol, max_iter, newton_step);

    // initialize fe_field_function
    dzyaloshinskii_function 
        = std::make_unique<dealii::Functions::FEFieldFunction<1>>
          (dzyaloshinskii_system->return_dof_handler(), 
           dzyaloshinskii_system->return_solution());
}




template <int dim>
double DzyaloshinskiiFunction<dim>::
value(const dealii::Point<dim> &p,
      const unsigned int component) const
{
    dealii::Point<1> theta( std::atan2(p[1] - defect_center[1], 
                                       p[0] - defect_center[0]) );
    theta[0] += 0 ? theta[0] >= 0 : 2 * M_PI;
    double phi = dzyaloshinskii_function->value(theta);

    double r = std::sqrt( (p[0] - defect_center[0])
                          *(p[0] - defect_center[0]) + 
                          (p[1] - defect_center[1])
                          *(p[1] - defect_center[1]) );
    double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
    double return_value = 0;

    switch (component)
    {
    case 0:
    	return_value = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
    	break;
    case 1:
    	return_value = 0.5 * S * std::sin(2*phi);
    	break;
    case 2:
    	return_value = 0.0;
    	break;
    case 3:
    	return_value = 0.5 * S * ( 1.0/3.0 - std::cos(2*phi) );
    	break;
    case 4:
    	return_value = 0.0;
    	break;
    }

    return return_value;
};



template <int dim>
void DzyaloshinskiiFunction<dim>::
vector_value(const dealii::Point<dim> &p,
			 dealii::Vector<double> &value) const
{
    dealii::Point<1> theta( std::atan2(p[1] - defect_center[1], 
                                       p[0] - defect_center[0]) );
    theta[0] += theta[0] >= 0 ? 0 : 2 * M_PI;
    double phi = dzyaloshinskii_function->value(theta);

    double r = std::sqrt( (p[0] - defect_center[0])
                          *(p[0] - defect_center[0]) + 
                          (p[1] - defect_center[1])
                          *(p[1] - defect_center[1]) );
    double S = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);

    value[0] = 0.5 * S * ( 1.0/3.0 + std::cos(2*phi) );
    value[1] = 0.5 * S * std::sin(2*phi);
    value[2] = 0.0;
    value[3] = 0.5 * S * ( 1.0/3.0 - std::cos(2*phi) );
    value[4] = 0.0;
};



template <int dim>
void DzyaloshinskiiFunction<dim>::
value_list(const std::vector<dealii::Point<dim>> &point_list,
           std::vector<double> &value_list,
           const unsigned int component) const
{
    assert(point_list.size() == value_list.size() 
           && "Point list and value list are different sizes");

    std::size_t n = point_list.size();
    std::vector<dealii::Point<1>> theta(n);
    std::vector<double> S(n);
    double r = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        theta[i][0] = std::atan2(point_list[i][1] - defect_center[1], 
                                 point_list[i][0] - defect_center[0]);
        theta[i][0] += theta[i][0] >= 0 ? 0 : 2 * M_PI;
        r = std::sqrt( (point_list[i][0] - defect_center[0])
                        *(point_list[i][0] - defect_center[0]) 
                        + 
                        (point_list[i][1] - defect_center[1])
                        *(point_list[i][1] - defect_center[1]) );
        S[i] = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
    }

    std::vector<double> phi(n);
    dzyaloshinskii_function->value_list(theta, phi);

    switch (component)
    {
    case 0:
        for (std::size_t i = 0; i < n; ++i)
    	{
    	    value_list[i] = 0.5 * S[i] * ( 1.0/3.0 + std::cos(2*phi[i]) );
    	}
    	break;
    case 1:
        for (std::size_t i = 0; i < n; ++i)
    	{
    	    value_list[i] = 0.5 * S[i] * std::sin(2*phi[i]);
    	}
    	break;
    case 2:
        for (std::size_t i = 0; i < n; ++i)
    	    value_list[i] = 0.0;
    	break;
    case 3:
        for (std::size_t i = 0; i < n; ++i)
    	{
    	    value_list[i] = 0.5 * S[i] * ( 1.0/3.0 - std::cos(2*phi[i]) );
    	}
    	break;
    case 4:
        for (std::size_t i = 0; i < n; ++i)
    	    value_list[i] = 0.0;
    	break;
    }
};



template <int dim>
void DzyaloshinskiiFunction<dim>::
vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                  std::vector<dealii::Vector<double>>   &value_list) const
{
    assert(point_list.size() == value_list.size() 
           && "Point list and value list are different sizes");

    std::size_t n = point_list.size();
    std::vector<dealii::Point<1>> theta(n);
    std::vector<double> S(n);
    double r = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        theta[i][0] = std::atan2(point_list[i][1] - defect_center[1], 
                                 point_list[i][0] - defect_center[0]);
        theta[i][0] += theta[i][0] >= 0 ? 0 : 2 * M_PI;
        r = std::sqrt( (point_list[i][0] - defect_center[0])
                        *(point_list[i][0] - defect_center[0]) 
                        + 
                        (point_list[i][1] - defect_center[1])
                        *(point_list[i][1] - defect_center[1]) );
        S[i] = S0 * (2.0 / (1 + std::exp(-r)) - 1.0);
    }

    std::vector<double> phi(n);
    dzyaloshinskii_function->value_list(theta, phi);

    for (std::size_t i = 0; i < n; ++i)
    {
        value_list[i][0] = 0.5 * S[i] * ( 1.0/3.0 + std::cos(2*phi[i]) );
        value_list[i][1] = 0.5 * S[i] * std::sin(2*phi[i]);
        value_list[i][2] = 0.0;
        value_list[i][3] = 0.5 * S[i] * ( 1.0/3.0 - std::cos(2*phi[i]) );
        value_list[i][4] = 0.0;
    }
};

template class DzyaloshinskiiFunction<3>;
template class DzyaloshinskiiFunction<2>;

BOOST_CLASS_EXPORT_IMPLEMENT(DzyaloshinskiiFunction<2>)
BOOST_CLASS_EXPORT_IMPLEMENT(DzyaloshinskiiFunction<3>)
