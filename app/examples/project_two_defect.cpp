#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/point.h>

#include <deal.II/numerics/data_out.h>

#include <vector>
#include <iostream>
#include <fstream>

#include "Utilities/maier_saupe_constants.hpp"
#include "ExampleFunctions/TensorToVector.hpp"
#include "ExampleFunctions/TwoDefectQTensor.hpp"
#include "Postprocessors/NematicPostprocessor.hpp"

namespace msc = maier_saupe_constants;

int main()
{
    const int dim = 2;

    const double left = -10.0;
    const double right = 10.0;
    const int num_refines = 9;

    std::vector<dealii::Point<dim>> centers(2);
    centers[0][0] = -5.0;
    centers[0][1] = 0;
    centers[1][0] = 5.0;
    centers[1][1] = 0;
    double r = 1.0;

    TensorToVector<dim, TwoDefectQTensor<dim>> function(centers, r);

    dealii::Triangulation<dim> tria;
    dealii::DoFHandler<dim, dim> dof_handler(tria);
    dealii::GridGenerator::hyper_cube(tria, left, right);
    tria.refine_global(num_refines);

    dealii::FESystem<dim> fe(dealii::FE_Q<dim>(1), msc::vec_dim<dim>);
    dof_handler.distribute_dofs(fe);

    dealii::Vector<double> solution(dof_handler.n_dofs());

    dealii::VectorTools::project(dof_handler,
                                 dealii::AffineConstraints<double>(),
                                 dealii::QGauss<dim>(fe.degree + 1),
                                 function,
                                 solution);

    NematicPostprocessor<dim> nematic_postprocessor;
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, nematic_postprocessor);
    data_out.build_patches();

    std::ofstream output("project_two_defect_output.vtu");
    data_out.write_vtu(output);

    return 0;
}
