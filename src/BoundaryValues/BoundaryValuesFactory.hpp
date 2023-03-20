#ifndef BOUNDARY_VALUES_FACTORY_HPP
#define BOUNDARY_VALUES_FACTORY_HPP

#include <deal.II/base/parameter_handler.h>

#include "BoundaryValues/MultiDefectConfiguration.hpp"
#include "BoundaryValuesInterface.hpp"
#include "BoundaryValues.hpp"
#include "DefectConfiguration.hpp"
#include "TwoDefectConfiguration.hpp"
#include "UniformConfiguration.hpp"
#include "PeriodicConfiguration.hpp"
#include "PeriodicSConfiguration.hpp"
#include "DzyaloshinskiiFunction.hpp"
#include "Utilities/ParameterParser.hpp"

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
                          dealii::Patterns::Selection("uniform"
                                                      "|periodic"
                                                      "|periodic-S"
                                                      "|defect|two-defect"
                                                      "|dzyaloshinskii-function"
                                                      "|multi-defect-configuration"),
                          "Name of inital condition + boundary value");
        prm.declare_entry("Boundary condition",
                          "Dirichlet",
                          dealii::Patterns::Selection("Dirichlet|Neumann"),
                          "Whether boundary condition is Dirichlet or Neumann");
        prm.declare_entry("S value",
                          "0.6751",
                          dealii::Patterns::Double(),
                          "Ambient S-value for any configuration");

        prm.enter_subsection("Defect configurations");
        prm.declare_entry("Defect positions",
                          "[0.0, 0.0]",
                          dealii::Patterns::Anything(),
                          "List of defect positions -- coordinates are comma "
                          "separated values in square brackets, points are "
                          "separated by spaces");
        prm.declare_entry("Defect charges",
                          "0.5",
                          dealii::Patterns::Anything(),
                          "List of defect charges -- charges are comma "
                          "separated values");
        prm.declare_entry("Defect orientations",
                          "0.5",
                          dealii::Patterns::Anything(),
                          "List of defect orientations -- charges are comma "
                          "separated values");
        prm.declare_entry("Defect radius",
                          "2.5",
                          dealii::Patterns::Double(),
                          "Radius around defects at which boundary is held "
                          "fixed for a multi-defect-configuration");
        prm.declare_entry("Defect charge name",
                          "plus-half",
                          dealii::Patterns::Selection("plus-half|minus-half"
                                                      "|plus-one|minus-one"
                                                      "|plus-half-minus-half"
                                                      "|plus-half-minus-half-alt"),
                          "Name of defect configuration");
        prm.leave_subsection();

        prm.enter_subsection("Dzyaloshinskii"); 
        prm.declare_entry("Anisotropy eps",
                          "0.0",
                          dealii::Patterns::Double(),
                          "Director anisotropy parameter value for "
                          "calculating Dzyaloshinskii solution");
        prm.declare_entry("Degree",
                          "1",
                          dealii::Patterns::Integer(),
                          "Degree of finite element scheme used to calculate "
                          "Dzyaloshinskii solution");
        prm.declare_entry("Charge",
                          "0.5",
                          dealii::Patterns::Double(),
                          "Charge of Dzyaloshinskii defect");
        prm.declare_entry("N refines",
                          "10",
                          dealii::Patterns::Integer(),
                          "Number of line refines for Dzyaloshinskii "
                          "numerical solution");
        prm.declare_entry("Tol",
                          "1e-10",
                          dealii::Patterns::Double(),
                          "Maximal residual for Newton's method when "
                          "calculating Dzyaloshinskii solution");
        prm.declare_entry("Max iter",
                          "100",
                          dealii::Patterns::Integer(),
                          "Maximal iterations for Newton's method when "
                          "calculating Dzyaloshinskii solution");
        prm.declare_entry("Newton step",
                          "1.0",
                          dealii::Patterns::Double(),
                          "Newton step size for calculating Dzyaloshinskii "
                          "solution");
        prm.leave_subsection();

        prm.enter_subsection("Periodic configurations");
        prm.declare_entry("Phi",
                          "0.0",
                          dealii::Patterns::Double(),
                          "Director angle for uniform configuration");
        prm.declare_entry("K",
                          "1.0",
                          dealii::Patterns::Double(),
                          "Wavenumber for periodic configurations");
        prm.declare_entry("Eps",
                          "0.1",
                          dealii::Patterns::Double(),
                          "Perturbation amplitude for periodic configurations");
        prm.leave_subsection();

        prm.leave_subsection();
    }



    template <int dim>
    std::map<std::string, boost::any>
    get_parameters(dealii::ParameterHandler &prm)
    {
        std::map<std::string, boost::any> bv_params;

        prm.enter_subsection("Boundary values");
        bv_params["boundary-values-name"] = prm.get("Name");
        bv_params["boundary-condition"] = prm.get("Boundary condition");
        bv_params["S-value"] = prm.get_double("S value");

        prm.enter_subsection("Defect configurations");
        bv_params["defect-positions"] 
            = ParameterParser::
              parse_coordinate_list<dim>(prm.get("Defect positions"));
        bv_params["defect-charges"]
            = ParameterParser::
              parse_number_list(prm.get("Defect charges"));
        bv_params["defect-orientations"]
            = ParameterParser::
              parse_number_list(prm.get("Defect orientations"));
        bv_params["defect-radius"] = prm.get_double("Defect radius");
        bv_params["defect-charge-name"] = prm.get("Defect charge name");
        prm.leave_subsection();

        prm.enter_subsection("Dzyaloshinskii");
        bv_params["anisotropy-eps"] = prm.get_double("Anisotropy eps");
        bv_params["degree"] = prm.get_integer("Degree");
        bv_params["charge"] = prm.get_double("Charge");
        bv_params["n-refines"] = prm.get_integer("N refines");
        bv_params["tol"] = prm.get_double("Tol");
        bv_params["max-iter"] = prm.get_integer("Max iter");
        bv_params["newton-step"] = prm.get_double("Newton step");
        prm.leave_subsection();

        prm.enter_subsection("Periodic configurations");
        bv_params["phi"] = prm.get_double("Phi");
        bv_params["k"] = prm.get_double("K");
        bv_params["eps"] = prm.get_double("Eps");
        prm.leave_subsection();

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
        else if (name == "periodic-S")
        {
            if (am.empty())
                return std::make_unique<PeriodicSConfiguration<dim>>();
            else
                return std::make_unique<PeriodicSConfiguration<dim>>(am);
        }
        else if (name == "dzyaloshinskii-function")
        {
            if (am.empty())
                return std::make_unique<DzyaloshinskiiFunction<dim>>();
            else
                return std::make_unique<DzyaloshinskiiFunction<dim>>(am);
        }
        else if (name == "multi-defect-configuration")
        {
            if (am.empty())
                return std::make_unique<MultiDefectConfiguration<dim>>();
            else
                return std::make_unique<MultiDefectConfiguration<dim>>(am);
        }
        else
        {
            throw std::invalid_argument("Invalid boundary value name in BoundaryValuesFactory");
        }
    }
} // namespace BoundaryValuesFactory

#endif
