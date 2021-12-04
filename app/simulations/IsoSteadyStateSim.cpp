// #include <deal.II/grid/tria.h>
// #include <deal.II/dofs/dof_handler.h>
// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/grid_out.h>

// #include <deal.II/fe/fe_system.h>
// #include <deal.II/fe/fe_q.h>

// #include <deal.II/dofs/dof_tools.h>

// #include <deal.II/fe/fe_values.h>
// #include <deal.II/base/quadrature_lib.h>

// #include <deal.II/base/function.h>
// #include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/matrix_tools.h>

// #include <deal.II/lac/vector.h>
// #include <deal.II/lac/full_matrix.h>
// #include <deal.II/lac/sparse_matrix.h>
// #include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/solver_bicgstab.h>
// #include <deal.II/lac/precondition.h>
// #include <deal.II/lac/affine_constraints.h>
// #include <deal.II/base/array_view.h>

// #include <deal.II/dofs/dof_renumbering.h>

// #include <deal.II/lac/sparse_direct.h>

// #include <deal.II/numerics/data_postprocessor.h>

// #include "maier_saupe_constants.hpp"
// #include "LagrangeMultiplier.hpp"
// #include "BoundaryValues/BoundaryValues.hpp"
// #include "BoundaryValues/DefectConfiguration.hpp"
// #include "BoundaryValues/BoundaryValuesInterface.hpp"
// #include "BoundaryValues/BoundaryValuesFactory.hpp"
// #include "Postprocessors/DirectorPostprocessor.hpp"
// #include "Postprocessors/SValuePostprocessor.hpp"

// #include <deal.II/numerics/data_out.h>

// #include <iostream>
// #include <fstream>
// #include <cmath>
// #include <chrono>
// #include <memory>

// #include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/text_iarchive.hpp>
#include "LiquidCrystalSystems/IsoSteadyState.hpp"
#include <boost/program_options.hpp>
#include <iostream>


// using namespace dealii;
// namespace po = boost::program_options;
// namespace msc = maier_saupe_constants;

// // TODO: make some of these parameters, read the rest in from maier-saupe-constants
// namespace {
// 	constexpr int order{590};
// }


// template <int dim>
// class IsoSteadyState
// {
// public:
// 	IsoSteadyState(const po::variables_map &vm);
// 	void run();
	
// private:
// 	void make_grid(const unsigned int num_refines,
//                  const double left = 0,
//                  const double right = 1);
// 	void setup_system(bool initial_step);
// 	void assemble_system();
// 	void solve();
// 	void set_boundary_values();
// 	double determine_step_length();

// 	void output_results(const std::string data_folder,
//                       const std::string filename) const;
// 	void save_data(const std::string data_folder,
//                  const std::string filename) const;
// 	void output_grid(const std::string data_folder,
//                    const std::string filename) const;
// 	void output_sparsity_pattern(const std::string data_folder,
//                                const std::string filename) const;

// 	Triangulation <dim> triangulation;
// 	DoFHandler<dim> dof_handler;
// 	FESystem<dim> fe;
// 	SparsityPattern sparsity_pattern;
// 	SparseMatrix<double> system_matrix;

// 	AffineConstraints<double> hanging_node_constraints;
//   std::unique_ptr<BoundaryValues<dim>> boundary_value_func;

//   Vector<double> current_solution;
//   Vector<double> system_update;
//   Vector<double> system_rhs;

//   LagrangeMultiplier<order> lagrange_multiplier;

//   double left_endpoint;
//   double right_endpoint;
//   double num_refines;

//   double simulation_step_size;
//   double simulation_tol;
//   int simulation_max_iters;
//   double maier_saupe_alpha;

//   std::string boundary_values_name;
//   double S_value;
//   std::string defect_charge_name;

//   std::string data_folder;
//   std::string initial_config_filename;
//   std::string final_config_filename;
//   std::string archive_filename;
// };



// // TODO: clean this class up, actually comment what's going on, then add
// // comments such that doxygen will generate a file walking someone through.
// template <int dim>
// IsoSteadyState<dim>::IsoSteadyState(const po::variables_map &vm)
// 	: dof_handler(triangulation)
// 	, fe(FE_Q<dim>(1), msc::vec_dim<dim>)
//   , boundary_value_func(BoundaryValuesFactory::BoundaryValuesFactory<dim>(vm))
//   , lagrange_multiplier(vm["lagrange-step-size"].as<double>(),
//                         vm["lagrange-tol"].as<double>(),
//                         vm["lagrange-max-iters"].as<int>())

//   , left_endpoint(vm["left-endpoint"].as<double>())
//   , right_endpoint(vm["right-endpoint"].as<double>())
//   , num_refines(vm["num-refines"].as<int>())

//   , simulation_step_size(vm["simulation-step-size"].as<double>())
//   , simulation_tol(vm["simulation-tol"].as<double>())
//   , simulation_max_iters(vm["simulation-max-iters"].as<int>())
//   , maier_saupe_alpha(vm["maier-saupe-alpha"].as<double>())

//   , boundary_values_name(vm["boundary-values-name"].as<std::string>())
//   , S_value(vm["S-value"].as<double>())
//   , defect_charge_name(vm["defect-charge-name"].as<std::string>())

//   , data_folder(vm["data-folder"].as<std::string>())
//   , initial_config_filename(vm["initial-config-filename"].as<std::string>())
//   , final_config_filename(vm["final-config-filename"].as<std::string>())
//   , archive_filename(vm["archive-filename"].as<std::string>())
// {}



// template <int dim>
// void IsoSteadyState<dim>::make_grid(const unsigned int num_refines,
//                                     const double left,
//                                     const double right)
// {
// 	GridGenerator::hyper_cube(triangulation, left, right);
// 	triangulation.refine_global(num_refines);
// }



// template <int dim>
// void IsoSteadyState<dim>::setup_system(bool initial_step)
// {
// 	if (initial_step) {
// 		dof_handler.distribute_dofs(fe);
// 		current_solution.reinit(dof_handler.n_dofs());

// 		hanging_node_constraints.clear();
// 		DoFTools::make_hanging_node_constraints(dof_handler,
// 												hanging_node_constraints);
// 		hanging_node_constraints.close();

// 		VectorTools::project(dof_handler,
// 							 hanging_node_constraints,
// 							 QGauss<dim>(fe.degree + 1),
// 							 *boundary_value_func,
// 							 current_solution);
// 	}
// 	system_update.reinit(dof_handler.n_dofs());
// 	system_rhs.reinit(dof_handler.n_dofs());

// 	DynamicSparsityPattern dsp(dof_handler.n_dofs());
// 	DoFTools::make_sparsity_pattern(dof_handler, dsp);

// 	hanging_node_constraints.condense(dsp);

// 	sparsity_pattern.copy_from(dsp);
// 	system_matrix.reinit(sparsity_pattern);
// }



// template <int dim>
// void IsoSteadyState<dim>::assemble_system()
// {
// 	QGauss<dim> quadrature_formula(fe.degree + 1);

// 	system_matrix = 0;
// 	system_rhs = 0;

// 	FEValues<dim> fe_values(fe,
// 							quadrature_formula,
// 							update_values
// 							| update_gradients
// 							| update_JxW_values);

// 	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
// 	const unsigned int n_q_points = quadrature_formula.size();

// 	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
// 	Vector<double> cell_rhs(dofs_per_cell);

// 	std::vector<std::vector<Tensor<1, dim>>>
// 	old_solution_gradients(n_q_points,
// 						   std::vector<Tensor<1, dim, double>>(fe.components));
// 	std::vector<Vector<double>>
// 	old_solution_values(n_q_points, Vector<double>(fe.components));
// 	Vector<double> Lambda(fe.components);
// 	LAPACKFullMatrix<double> R(fe.components, fe.components);
// 	std::vector<Vector<double>> R_inv_phi(dofs_per_cell,
// 										  Vector<double>(fe.components));

// 	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

// 	for (const auto &cell : dof_handler.active_cell_iterators())
// 	{
// 		cell_matrix = 0;
// 		cell_rhs = 0;

// 		fe_values.reinit(cell);
// 		fe_values.get_function_gradients(current_solution,
//                                      old_solution_gradients);
// 		fe_values.get_function_values(current_solution,
//                                   old_solution_values);

// 		for (unsigned int q = 0; q < n_q_points; ++q)
// 		{
// 			Lambda.reinit(fe.components);
// 			R.reinit(fe.components);

// 			lagrange_multiplier.invertQ(old_solution_values[q]);
// 			lagrange_multiplier.returnLambda(Lambda);
// 			lagrange_multiplier.returnJac(R);
// 			R.compute_lu_factorization();

// 			for (unsigned int j = 0; j < dofs_per_cell; ++j)
// 			{
// 				const unsigned int component_j =
// 						fe.system_to_component_index(j).first;
// 				R_inv_phi[j].reinit(fe.components);
// 				R_inv_phi[j][component_j] = fe_values.shape_value(j, q);
// 				R.solve(R_inv_phi[j]);
// 			}

// 			for (unsigned int i = 0; i < dofs_per_cell; ++i)
// 			{
// 				const unsigned int component_i =
// 					fe.system_to_component_index(i).first;

// 				for (unsigned int j = 0; j < dofs_per_cell; ++j)
// 				{
// 					const unsigned int component_j =
// 						fe.system_to_component_index(j).first;

// 					cell_matrix(i, j) +=
// 						(((component_i == component_j) ?
// 						  (fe_values.shape_value(i, q)
// 						  * maier_saupe_alpha
// 						  * fe_values.shape_value(j, q)) :
// 						  0)
// 						 -
// 						 ((component_i == component_j) ?
// 						 (fe_values.shape_grad(i, q)
// 						  * fe_values.shape_grad(j, q)) :
// 						  0)
// 						 -
// 						 (fe_values.shape_value(i, q)
// 						  * R_inv_phi[j][component_i]))
// 						 * fe_values.JxW(q);
// 				}
// 				cell_rhs(i) +=
// 					(-(fe_values.shape_value(i, q)
// 					   * maier_saupe_alpha
// 					   * old_solution_values[q][component_i])
// 					  +
// 					  (fe_values.shape_grad(i, q)
// 					   * old_solution_gradients[q][component_i])
// 					  +
// 					  (fe_values.shape_value(i, q)
// 					   * Lambda[component_i]))
// 					  * fe_values.JxW(q);
// 			}
// 		}

// 		cell->get_dof_indices(local_dof_indices);
// 		for (unsigned int i = 0; i < dofs_per_cell; ++i)
// 		{
// 			for (unsigned int j = 0; j < dofs_per_cell; ++j)
// 			{
// 				system_matrix.add(local_dof_indices[i],
// 								  local_dof_indices[j],
// 								  cell_matrix(i, j));
// 			}
// 			system_rhs(local_dof_indices[i]) += cell_rhs(i);
// 		}
// 	}

// 	hanging_node_constraints.condense(system_matrix);
// 	hanging_node_constraints.condense(system_rhs);

// 	std::map<types::global_dof_index, double> boundary_values;
// 	VectorTools::interpolate_boundary_values(dof_handler,
//                                            0,
//                                            Functions::ZeroFunction<dim>(msc::vec_dim<dim>),
//                                            boundary_values);
// 	MatrixTools::apply_boundary_values(boundary_values,
//                                      system_matrix,
//                                      system_update,
//                                      system_rhs);
// }

// template <int dim>
// void IsoSteadyState<dim>::solve()
// {
// 	SparseDirectUMFPACK solver;
// 	solver.factorize(system_matrix);
// 	system_update = system_rhs;
// 	solver.solve(system_update);

// 	const double newton_alpha = determine_step_length();
// 	current_solution.add(newton_alpha, system_update);
// }



// template <int dim>
// void IsoSteadyState<dim>::set_boundary_values()
// {
// 	std::map<types::global_dof_index, double> boundary_values;
// 	VectorTools::interpolate_boundary_values(dof_handler,
//                                            0,
//                                            *boundary_value_func,
//                                            boundary_values);
// 	for (auto &boundary_value : boundary_values)
// 		current_solution(boundary_value.first) = boundary_value.second;

// 	hanging_node_constraints.distribute(current_solution);
// }



// template <int dim>
// double IsoSteadyState<dim>::determine_step_length()
// {
// 	return simulation_step_size;
// }



// template <int dim>
// void IsoSteadyState<dim>::output_grid(const std::string folder,
//                                       const std::string filename) const
// {
// 	std::ofstream out(filename);
// 	GridOut grid_out;
// 	grid_out.write_svg(triangulation, out);
// 	std::cout << "Grid written to " << filename << std::endl;
// }



// template <int dim>
// void IsoSteadyState<dim>::output_sparsity_pattern
// (const std::string folder, const std::string filename) const
// {
// 	std::ofstream out(folder + filename);
// 	sparsity_pattern.print_svg(out);
// }




// template <int dim>
// void IsoSteadyState<dim>::output_results(const std::string folder,
//                                          const std::string filename) const
// {
// 	DirectorPostprocessor<dim>
//     director_postprocessor_defect(boundary_values_name);
// 	SValuePostprocessor<dim>
//     S_value_postprocessor_defect(boundary_values_name);
// 	DataOut<dim> data_out;

// 	data_out.attach_dof_handler(dof_handler);
// 	data_out.add_data_vector(current_solution, director_postprocessor_defect);
// 	data_out.add_data_vector(current_solution, S_value_postprocessor_defect);
// 	data_out.build_patches();

// 	std::cout << "Outputting results" << std::endl;

// 	std::ofstream output(folder + filename);
// 	data_out.write_vtu(output);
// }



// // TODO Save the rest of the class parameters from IsoSteadyState
// template <int dim>
// void IsoSteadyState<dim>::save_data(const std::string folder,
//                                     const std::string filename) const
// {
// 	std::ofstream ofs(folder + filename);
// 	boost::archive::text_oarchive oa(ofs);

// 	current_solution.save(oa, 1);
// 	dof_handler.save(oa, 1);
// }




// template <int dim>
// void IsoSteadyState<dim>::run()
// {
// 	make_grid(num_refines,
//             left_endpoint,
//             right_endpoint);

// 	setup_system(true);
// 	set_boundary_values();
// 	output_results(data_folder, initial_config_filename);

// 	unsigned int iterations = 0;
// 	double residual_norm{std::numeric_limits<double>::max()};
// 	auto start = std::chrono::high_resolution_clock::now();
// 	while (residual_norm > simulation_tol
//          && iterations < simulation_max_iters)
// 	{
// 		assemble_system();
// 		solve();
// 		residual_norm = system_rhs.l2_norm();
// 		std::cout << "Residual is: " << residual_norm << std::endl;
// 		std::cout << "Norm of newton update is: "
//               << system_update.l2_norm() << std::endl;
// 	}
// 	auto stop = std::chrono::high_resolution_clock::now();

// 	auto duration =
// 		std::chrono::duration_cast<std::chrono::seconds>(stop - start);
// 	std::cout << "total time for solving is: "
//             << duration.count() << " seconds" << std::endl;

// 	output_results(data_folder, final_config_filename);
// 	save_data(data_folder, archive_filename);
// }

namespace po = boost::program_options;

int main(int ac, char* av[])
{
  // TODO: Figure out how to set dim and order at runtime
    po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")

    // Set BoundaryValues parameters
    ("boundary-values-name",
     po::value<std::string>()->default_value("defect"),
     "sets boundary value scheme")
    ("S-value", po::value<double>()->default_value(0.6751),
     "sets S value at the boundaries")
    ("defect-charge-name",
     po::value<std::string>()->default_value("plus-half"),
     "sets defect charge of initial configuration")

    // Set LagrangeMultiplier parameters
    ("lagrange-step-size", po::value<double>()->default_value(1.0),
     "step size of Newton's method for Lagrange Multiplier scheme")
    ("lagrange-max-iters", po::value<int>()->default_value(20),
     "maximum iterations for Newton's method in Lagrange Multiplier scheme")
    ("lagrange-tol", po::value<double>()->default_value(1e-8),
     "tolerance of squared norm in Lagrange Multiplier scheme")

    // Set domain parameters
    ("left-endpoint", po::value<double>()->default_value(-10 / std::sqrt(2)),
     "left endpoint of square domain grid")
    ("right-endpoint", po::value<double>()->default_value(10 / std::sqrt(2)),
     "right endpoint of square domain grid")
    ("num-refines", po::value<int>()->default_value(4),
     "number of times to refine domain grid")

    // Set simulation Newton's method parameters
    ("simulation-step-size", po::value<double>()->default_value(1.0),
     "step size for simulation-level Newton's method")
    ("simulation-tol", po::value<double>()->default_value(1e-8),
     "tolerance of normed residual for simulation-level Newton's method")
    ("simulation-max-iters", po::value<int>()->default_value(10),
     "maximum iterations for simulation-level Newton's method")
    ("maier-saupe-alpha", po::value<double>()->default_value(8.0),
     "alpha constant in Maier-Saupe free energy")

    // Set data output parameters
    ("data-folder",
     po::value<std::string>()->default_value("./"),
     "path to folder where output data will be saved")
    ("initial-config-filename",
     po::value<std::string>()->default_value("initial-configuration.vtu"),
     "filename of initial configuration data")
    ("final-config-filename",
     po::value<std::string>()->default_value("final-configuration.vtu"),
     "filename of final configuration data")
    ("archive-filename",
     po::value<std::string>()->default_value("iso-steady-state.dat"),
     "filename of archive of IsoSteadyState class")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
    {
      std::cout << desc << "\n";
      return 0;
    }

	const int dim = 2;
  const int order = 590;
	// IsoSteadyState<dim> iso_steady_state(vm);
  IsoSteadyState<dim, order> iso_steady_state(vm);
  iso_steady_state.run();

	return 0;
}
