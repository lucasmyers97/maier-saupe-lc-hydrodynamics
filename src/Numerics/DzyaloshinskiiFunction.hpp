#ifndef DZYALOSHINSKII_FUNCTION_HPP
#define DZYALOSHINSKII_FUNCTION_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/fe_field_function.h>

#include <memory>
#include <cmath>

#include "Utilities/maier_saupe_constants.hpp"
#include "LiquidCrystalSystems/DzyaloshinskiiSystem.hpp"

template <int dim>
class DzyaloshinskiiFunction : public dealii::Function<dim>
{
public:
    DzyaloshinskiiFunction(const dealii::Point<dim> &p, double S0_)
        : dealii::Function<dim>(maier_saupe_constants::vec_dim<dim>)
        , defect_center(p)
        , S0(S0_)
    {}

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        dealii::Point<1> theta( std::atan2(p[1] - defect_center[1], 
                                           p[0] - defect_center[0]) );
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

    virtual void vector_value(const dealii::Point<dim> &p,
					          dealii::Vector<double> &value) const override
    {
        dealii::Point<1> theta( std::atan2(p[1] - defect_center[1], 
                                           p[0] - defect_center[0]) );
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

    virtual void value_list(const std::vector<dealii::Point<dim>> &point_list,
                            std::vector<double> &value_list,
                            const unsigned int component = 0) const override
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

    virtual void
    vector_value_list(const std::vector<dealii::Point<dim>> &point_list,
                      std::vector<dealii::Vector<double>>   &value_list)
                      const override
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

    void initialize(double eps,
                    unsigned int degree,
                    double charge,
                    unsigned int n_refines, 
                    double tol, 
                    unsigned int max_iter, 
                    double newton_step)
    {
        dzyaloshinskii_system
            = std::make_unique<DzyaloshinskiiSystem>(eps, degree, charge);

        // run dzyaloshinskii sim
        dzyaloshinskii_system->make_grid(n_refines);
        dzyaloshinskii_system->setup_system();
        dzyaloshinskii_system->run_newton_method(tol, max_iter, newton_step);

        // initialize fe_field_function
        dzyaloshinskii_function 
            = std::make_unique<dealii::Functions::FEFieldFunction<1>>
              (dzyaloshinskii_system->return_dof_handler(), 
               dzyaloshinskii_system->return_solution());
    };

private:
    dealii::Point<dim> defect_center;
    double S0;

    std::unique_ptr<DzyaloshinskiiSystem> dzyaloshinskii_system;
    std::unique_ptr<dealii::Functions::FEFieldFunction<1>> 
        dzyaloshinskii_function;
};

#endif
