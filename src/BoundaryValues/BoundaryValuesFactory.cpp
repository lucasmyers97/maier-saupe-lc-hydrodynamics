#include "BoundaryValues/BoundaryValuesFactory.hpp"

#include <deal.II/base/parameter_handler.h>

#include "BoundaryValues/MultiDefectConfiguration.hpp"
#include "BoundaryValues/PerturbativeTwoDefect.hpp"
#include "BoundaryValuesInterface.hpp"
#include "BoundaryValues.hpp"
#include "DefectConfiguration.hpp"
#include "TwoDefectConfiguration.hpp"
#include "TwistedTwoDefect.hpp"
#include "UniformConfiguration.hpp"
#include "PeriodicConfiguration.hpp"
#include "PeriodicSConfiguration.hpp"
#include "DzyaloshinskiiFunction.hpp"
#include "EscapedRadial.hpp"
#include "Utilities/ParameterParser.hpp"
#include "Utilities/vector_conversion.hpp"

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
                                                      "|multi-defect-configuration"
                                                      "|perturbative-two-defect"),
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

        prm.enter_subsection("Perturbative two defect");
        prm.declare_entry("Defect distance",
                          "10.0",
                          dealii::Patterns::Double(),
                          "Distance between two defects");
        prm.declare_entry("Defect position name",
                          "left",
                          dealii::Patterns::Selection("left|right"),
                          "Whether current defect is on the left or right");
        prm.declare_entry("Defect isomorph name",
                          "a",
                          dealii::Patterns::Selection("a|b"),
                          "Which of the two Zumer isomorphs the configuration is");
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

        prm.enter_subsection("Perturbative two defect");
        bv_params["defect-distance"] = prm.get_double("Defect distance");
        bv_params["defect-position-name"] = prm.get("Defect position name");
        bv_params["defect-isomorph-name"] = prm.get("Defect isomorph name");
        prm.leave_subsection();

        prm.leave_subsection();

        return bv_params;
    }


    template <int dim>
    std::map<std::string, boost::any>
    parse_parameters(const toml::table& bv_table)
    {
        std::map<std::string, boost::any> bv_params;

        const auto name = bv_table["name"].value<std::string>();
        const auto boundary_condition = bv_table["boundary_condition"].value<std::string>();
        const auto S_value = bv_table["S_value"].value<double>();

        if (!bv_table["defect_configurations"].is_table())
            throw std::invalid_argument("No defect_configurations table in toml file");

        const toml::table& defect_config_table = *bv_table["defect_configurations"].as_table();

        if (!defect_config_table["defect_positions"].is_array())
            throw std::invalid_argument("No defect_positions array in toml file");

        const auto defect_positions 
            = toml::convert<std::vector<std::vector<double>>>(
                        *defect_config_table["defect_positions"].as_array()
                        );

        if (!defect_config_table["defect_charges"].is_array())
            throw std::invalid_argument("No defect_charges array in toml file");
        const auto defect_charges
            = toml::convert<std::vector<double>>(
                        *defect_config_table["defect_charges"].as_array()
                        );

        if (!defect_config_table["defect_orientations"].is_array())
            throw std::invalid_argument("No defect_orientations array in toml file");
        const auto defect_orientations
            = toml::convert<std::vector<double>>(
                        *defect_config_table["defect_orientations"].as_array()
                        );

        const auto defect_radius = defect_config_table["defect_radius"].value<double>();
        const auto defect_axis = defect_config_table["defect_axis"].value<std::string>();
        const auto defect_charge_name = defect_config_table["defect_charge_name"].value<std::string>();
        const auto twist_angular_speed = defect_config_table["twist_angular_speed"].value<double>();

        if (!bv_table["dzyaloshinskii"].is_table())
            throw std::invalid_argument("No dzyaloshinskii table in toml file");
        const toml::table& dzyaloshinskii_table = *bv_table["dzyaloshinskii"].as_table();

        const auto anisotropy_eps = dzyaloshinskii_table["anisotropy_eps"].value<double>();
        const auto degree = dzyaloshinskii_table["degree"].value<unsigned int>();
        const auto charge = dzyaloshinskii_table["charge"].value<double>();
        const auto n_refines = dzyaloshinskii_table["n_refines"].value<unsigned int>();
        const auto tol = dzyaloshinskii_table["tol"].value<double>();
        const auto max_iter = dzyaloshinskii_table["max_iter"].value<unsigned int>();
        const auto newton_step = dzyaloshinskii_table["newton_step"].value<double>();

        if (!bv_table["periodic_configurations"].is_table())
            throw std::invalid_argument("No periodic_configurations table in toml file");
        const toml::table& periodic_table = *bv_table["periodic_configurations"].as_table();

        const auto phi = periodic_table["phi"].value<double>();
        const auto k = periodic_table["k"].value<double>();
        const auto eps = periodic_table["eps"].value<double>();

        if (!bv_table["perturbative_two_defect"].is_table())
            throw std::invalid_argument("No perturbative_two_defect table in toml file");
        const toml::table& perturbative_table = *bv_table["perturbative_two_defect"].as_table();

        const auto defect_distance = perturbative_table["defect_distance"].value<double>();
        const auto defect_position_name = perturbative_table["defect_position_name"].value<std::string>();
        const auto defect_isomorph_name = perturbative_table["defect_isomorph_name"].value<std::string>();

        if (!bv_table["escaped_radial"].is_table())
            throw std::invalid_argument("No escaped_radial table in toml file");
        const toml::table& er_table = *bv_table["escaped_radial"].as_table();

        const auto er_cylinder_radius = er_table["cylinder_radius"].value<double>();
        if (!er_table["center_axis"].is_array())
            throw std::invalid_argument("No center_axis array in toml file");
        const auto er_center_axis
            = toml::convert<std::vector<double>>(*er_table["center_axis"].as_array());
        const auto er_axis = er_table["axis"].value<std::string>();

        if (!name) throw std::invalid_argument("No boundary_values name in parameter file");
        if (!boundary_condition) throw std::invalid_argument("No boundary_values boundary_condition in parameter file");
        if (!S_value) throw std::invalid_argument("No boundary_values S_value in parameter file");

        if (!defect_radius) throw std::invalid_argument("No boundary_values defect_radius in parameter file");
        if (!defect_axis) throw std::invalid_argument("No boundary_values defect_axis in parameter file");
        if (!defect_charge_name) throw std::invalid_argument("No boundary_values defect_charge_name in parameter file");
        if (!twist_angular_speed) throw std::invalid_argument("No boundary_values twist_angular_speed in parameter file");

        if (!anisotropy_eps) throw std::invalid_argument("No boundary_values anisotropy_eps in parameter file");
        if (!degree) throw std::invalid_argument("No boundary_values degree in parameter file");
        if (!charge) throw std::invalid_argument("No boundary_values charge in parameter file");
        if (!n_refines) throw std::invalid_argument("No boundary_values n_refines in parameter file");
        if (!tol) throw std::invalid_argument("No boundary_values tol in parameter file");
        if (!max_iter) throw std::invalid_argument("No boundary_values max_iter in parameter file");
        if (!newton_step) throw std::invalid_argument("No boundary_values newton_step in parameter file");

        if (!phi) throw std::invalid_argument("No boundary_values phi in parameter file");
        if (!k) throw std::invalid_argument("No boundary_values k in parameter file");
        if (!eps) throw std::invalid_argument("No boundary_values eps in parameter file");

        if (!defect_distance) throw std::invalid_argument("No boundary_values defect_distance in parameter file");
        if (!defect_position_name) throw std::invalid_argument("No boundary_values defect_position_name in parameter file");
        if (!defect_isomorph_name) throw std::invalid_argument("No boundary_values defect_isomorph_name in parameter file");

        if (!er_cylinder_radius) throw std::invalid_argument("No cylinder_radius in parameter file");
        if (!er_axis) throw std::invalid_argument("No axis in parameter file");

        bv_params["boundary-values-name"] = name.value();
        bv_params["boundary-condition"] = boundary_condition.value();
        bv_params["S-value"] = S_value.value();

        bv_params["defect-positions"] = defect_positions;
        bv_params["defect-charges"] = defect_charges;
        bv_params["defect-orientations"] = defect_orientations;
        bv_params["defect-radius"] = defect_radius.value();
        bv_params["defect-axis"] = defect_axis.value();
        bv_params["defect-charge-name"] = defect_charge_name.value();
        bv_params["twist-angular-speed"] = twist_angular_speed.value();

        bv_params["anisotropy-eps"] = anisotropy_eps.value();
        bv_params["degree"] = degree.value();
        bv_params["charge"] = charge.value();
        bv_params["n-refines"] = n_refines.value();
        bv_params["tol"] = tol.value();
        bv_params["max-iter"] = max_iter.value();
        bv_params["newton-step"] = newton_step.value();

        bv_params["phi"] = phi.value();
        bv_params["k"] = k.value();
        bv_params["eps"] = eps.value();

        bv_params["defect-distance"] = defect_distance.value();
        bv_params["defect-position-name"] = defect_position_name.value();
        bv_params["defect-isomorph-name"] = defect_isomorph_name.value();

        bv_params["cylinder-radius"] = er_cylinder_radius.value();
        bv_params["center-axis"] = er_center_axis;
        bv_params["axis"] = er_axis;

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
        else if (name == "twisted-two-defect")
        {
            if (am.empty())
              return std::make_unique<TwistedTwoDefect<dim>>();
            else
                return std::make_unique<TwistedTwoDefect<dim>>(am);
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
        else if (name == "perturbative-two-defect")
        {
            if (am.empty())
                return std::make_unique<PerturbativeTwoDefect<dim>>();
            else
                return std::make_unique<PerturbativeTwoDefect<dim>>(am);
        }
        else if (name == "escaped-radial")
        {
            if (am.empty())
                return std::make_unique<EscapedRadial<dim>>();
            else
                return std::make_unique<EscapedRadial<dim>>(am);
        }
        else
        {
            throw std::invalid_argument("Invalid boundary value name in BoundaryValuesFactory");
        }
    }

    template
    void declare_parameters<2>(dealii::ParameterHandler &prm);

    template
    void declare_parameters<3>(dealii::ParameterHandler &prm);
    
    template
    std::map<std::string, boost::any> 
    get_parameters<2>(dealii::ParameterHandler &prm);

    template
    std::map<std::string, boost::any> 
    get_parameters<3>(dealii::ParameterHandler &prm);

    template
    std::map<std::string, boost::any>
    parse_parameters<2>(const toml::table& table);

    template
    std::map<std::string, boost::any>
    parse_parameters<3>(const toml::table& table);

    template
    std::unique_ptr<BoundaryValues<2>>
    BoundaryValuesFactory(const po::variables_map &vm);

    template
    std::unique_ptr<BoundaryValues<3>>
    BoundaryValuesFactory(const po::variables_map &vm);

    template
    std::unique_ptr<BoundaryValues<2>>
    BoundaryValuesFactory(std::map<std::string, boost::any> am);

    template
    std::unique_ptr<BoundaryValues<3>>
    BoundaryValuesFactory(std::map<std::string, boost::any> am);

} // namespace BoundaryValuesFactory
