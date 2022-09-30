#ifndef BOUNDARY_VALUES_FACTORY_HPP
#define BOUNDARY_VALUES_FACTORY_HPP

#include <deal.II/base/parameter_handler.h>

#include "BoundaryValuesInterface.hpp"
#include "BoundaryValues.hpp"
#include "DefectConfiguration.hpp"
#include "TwoDefectConfiguration.hpp"
#include "UniformConfiguration.hpp"
#include "PeriodicConfiguration.hpp"

#include <boost/program_options.hpp>
#include <boost/any.hpp>

#include <map>
#include <memory>
#include <string>
#include <stdexcept>

namespace BoundaryValuesFactory
{
    namespace po = boost::program_options;

    template <int dim>
    void declare_parameters(dealii::ParameterHandler &prm)
    {
        prm.enter_subsection("Boundary values");
        prm.declare_entry("Name",
                          "uniform",
                          dealii::Patterns::Selection("uniform|periodic"
                                                      "|defect|two-defect"));
        prm.declare_entry("S value",
                          "0.6751",
                          dealii::Patterns::Double());
        prm.declare_entry("Phi",
                          "0.0",
                          dealii::Patterns::Double());
        prm.declare_entry("K",
                          "1.0",
                          dealii::Patterns::Double());
        prm.declare_entry("Eps",
                          "0.1",
                          dealii::Patterns::Double());
        prm.declare_entry("Defect charge name",
                          "plus-half",
                          dealii::Patterns::Selection("plus-half|minus-half"
                                                      "|plus-one|minus-one"
                                                      "|plus-half-minus-half"
                                                      "|plus-half-minus-half-alt"));
        prm.declare_entry("Center x1",
                          "5.0",
                          dealii::Patterns::Double());
        prm.declare_entry("Center y1",
                          "0.0",
                          dealii::Patterns::Double());
        prm.declare_entry("Center x2",
                          "-5.0",
                          dealii::Patterns::Double());
        prm.declare_entry("Center y2",
                          "0.0",
                          dealii::Patterns::Double());
        prm.leave_subsection();
    }



    template <int dim>
    std::map<std::string, boost::any>
    get_parameters(dealii::ParameterHandler &prm)
    {
        prm.enter_subsection("Boundary values");
        std::map<std::string, boost::any> bv_params;
        bv_params["boundary-values-name"] = prm.get("Name");
        bv_params["S-value"] = prm.get_double("S value");
        bv_params["phi"] = prm.get_double("Phi");
        bv_params["k"] = prm.get_double("K");
        bv_params["eps"] = prm.get_double("Eps");
        bv_params["defect-charge-name"] = prm.get("Defect charge name");

        double x1 = prm.get_double("Center x1");
        double y1 = prm.get_double("Center y1");
        double x2 = prm.get_double("Center x2");
        double y2 = prm.get_double("Center y2");
        bv_params["centers"] = std::vector<double>({x1, y1, x2, y2});

        prm.leave_subsection();

        return bv_params;
    }

    template <int dim>
    std::unique_ptr<BoundaryValues<dim>>
    BoundaryValuesFactory(const po::variables_map &vm)
    {
        std::string name = vm["boundary-values-name"].as<std::string>();

        if (name == "uniform")
        {
            return std::make_unique<UniformConfiguration<dim>>(vm);
        }
        else if (name == "defect")
        {
            return std::make_unique<DefectConfiguration<dim>>(vm);
        }
        else if (name == "two-defect")
        {
            return std::make_unique<TwoDefectConfiguration<dim>>(vm);
        }
        else
        {
            throw std::invalid_argument("Invalid boundary value name in BoundaryValuesFactory");
        }
    }


    template <int dim>
    std::unique_ptr<BoundaryValues<dim>>
    BoundaryValuesFactory(std::map<std::string, boost::any> am)
    {
        std::string name 
            = boost::any_cast<std::string>(am["boundary-values-name"]);

        if (name == "uniform")
        {
            if (am.empty())
                return std::make_unique<UniformConfiguration<dim>>();
            else
                return std::make_unique<UniformConfiguration<dim>>(am);
        }
        else if (name == "defect")
        {
            if (am.empty())
                return std::make_unique<DefectConfiguration<dim>>();
            else
                return std::make_unique<DefectConfiguration<dim>>(am);
        }
        else if (name == "two-defect")
        {
            if (am.empty())
              return std::make_unique<TwoDefectConfiguration<dim>>();
            else
                return std::make_unique<TwoDefectConfiguration<dim>>(am);
        }
        else if (name == "periodic")
        {
            if (am.empty())
                return std::make_unique<PeriodicConfiguration<dim>>();
            else
                return std::make_unique<PeriodicConfiguration<dim>>(am);
        }
        else
        {
            throw std::invalid_argument("Invalid boundary value name in BoundaryValuesFactory");
        }
    }
} // namespace BoundaryValuesFactory

#endif
