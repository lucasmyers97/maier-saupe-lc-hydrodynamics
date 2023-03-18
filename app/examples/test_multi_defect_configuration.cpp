#include "BoundaryValues/MultiDefectConfiguration.hpp"
#include "Utilities/ParameterParser.hpp"
#include "Utilities/maier_saupe_constants.hpp"
#include "Postprocessors/NematicPostprocessor.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/data_out_base.h>

#include <boost/any.hpp>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>

namespace msc = maier_saupe_constants;
double left = -20.0;
double right = 20.0;
unsigned int n_refines = 8;

int main()
{
    constexpr int dim = 2;

    std::map<std::string, boost::any> am;
    am["boundary-condition"] = std::string("Dirichlet");
    am["S-value"] = 0.6751;
    am["defect-positions"] = ParameterParser::
                             parse_coordinate_list<dim>("[-10.0, 0], [10.0, 0]");
    am["defect-charges"] = ParameterParser::parse_number_list("0.5, -0.5");
    am["defect-orientations"] = ParameterParser::parse_number_list("0.0, 1.5707963267948");
    am["defect-radius"] = 3.0;
    am["anisotropy-eps"] = 0.5;
    am["degree"] = static_cast<long>(1);
    am["n-refines"] = static_cast<long>(10);
    am["tol"] = 1e-10;
    am["max-iter"] = static_cast<long>(100);
    am["newton-step"] = 1.0;

    boost::any_cast<std::string>(am["boundary-condition"]);
    boost::any_cast<double>(am["S-value"]);
    ParameterParser::vector_to_dealii_point<dim>( 
                boost::any_cast<std::vector<std::vector<double>>>(am["defect-positions"])
                );
    boost::any_cast<std::vector<double>>(am["defect-charges"]);
    boost::any_cast<std::vector<double>>(am["defect-orientations"]);
    boost::any_cast<double>(am["defect-radius"]);
    boost::any_cast<double>(am["anisotropy-eps"]);
    boost::any_cast<long>(am["degree"]);
    boost::any_cast<long>(am["n-refines"]);
    boost::any_cast<double>(am["tol"]);
    boost::any_cast<long>(am["max-iter"]);
    boost::any_cast<double>(am["newton-step"]);

    MultiDefectConfiguration<dim> multi_defect_configuration(am);

    dealii::Triangulation<dim> tria;
    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(n_refines);
    dealii::DoFHandler<dim> dof_handler(tria);
    dealii::FESystem<dim> fe(dealii::FE_Q<dim>(1), msc::vec_dim<dim>);
    dealii::AffineConstraints<double> constraints;
    dealii::Vector<double> solution;

    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());

    dealii::VectorTools::interpolate(dof_handler, 
                                     multi_defect_configuration, 
                                     solution);

    NematicPostprocessor<dim> nematic_postprocessor;
    dealii::DataOut<dim> data_out;
    dealii::DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, nematic_postprocessor);
    data_out.build_patches();

    std::ofstream output("multi_defect_configuration.vtu");
    data_out.write_vtu(output);

    return 0;
}
