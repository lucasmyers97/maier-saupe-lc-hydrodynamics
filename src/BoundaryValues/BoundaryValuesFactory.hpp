#ifndef BOUNDARY_VALUES_FACTORY_HPP
#define BOUNDARY_VALUES_FACTORY_HPP

#include <deal.II/base/parameter_handler.h>

#include "BoundaryValues/MultiDefectConfiguration.hpp"
#include "BoundaryValues/PerturbativeTwoDefect.hpp"
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
    void declare_parameters(dealii::ParameterHandler &prm);

    template <int dim>
    std::map<std::string, boost::any>
    get_parameters(dealii::ParameterHandler &prm);

    template <int dim>
    std::unique_ptr<BoundaryValues<dim>>
    BoundaryValuesFactory(const po::variables_map &vm);

    template <int dim>
    std::unique_ptr<BoundaryValues<dim>>
    BoundaryValuesFactory(std::map<std::string, boost::any> am);

} // namespace BoundaryValuesFactory

#endif
