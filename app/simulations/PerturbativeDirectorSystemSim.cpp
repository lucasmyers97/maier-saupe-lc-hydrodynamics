#include "LiquidCrystalSystems/PerturbativeDirectorSystem.cpp"
#include "LiquidCrystalSystems/PerturbativeDirectorSystem.hpp"
#include "Parameters/toml.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <memory>
#include <string>

int main(int argc, char *argv[])
{
    if (argc - 1 != 1)
        throw std::invalid_argument("Error! Didn't input filename");
    std::string toml_filename(argv[1]);

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    constexpr int dim = 2;

    // Get inputs from toml file
    const toml::table tbl = toml::parse_file(toml_filename);
    if (!tbl["perturbative_director_system"].is_table())
        throw std::invalid_argument("No perturbative_director_system table in toml file");
    const toml::table& pds_tbl = *tbl["perturbative_director_system"].as_table();

    if (!pds_tbl["simulation"].is_table())
        throw std::invalid_argument("No perturbative_director_system.simulation table in toml file");
    const toml::table& sim_tbl = *pds_tbl["simulation"].as_table();

    const auto degree = sim_tbl["degree"].value<unsigned int>();
    const auto solver_type_arg = sim_tbl["solver_type"].value<std::string>();
    const auto eps = sim_tbl["eps"].value<double>();
    if (!sim_tbl["defect_charges"].is_array())
        throw std::invalid_argument("No defect_charges array in toml file");
    const auto defect_charges
        = toml::convert<std::vector<double>>(*sim_tbl["defect_charges"].as_array());

    if (!pds_tbl["grid"].is_table())
        throw std::invalid_argument("No perturbative_director_system.grid table in toml file");
    const toml::table& grid_tbl = *pds_tbl["grid"].as_table();

    const auto left = grid_tbl["left"].value<double>();
    const auto right = grid_tbl["right"].value<double>();
    const auto grid_name = grid_tbl["grid_name"].value<std::string>();
    const auto grid_parameters = grid_tbl["grid_parameters"].value<std::string>();
    const auto num_refines = grid_tbl["num_refines"].value<unsigned int>();
    const auto num_further_refines = grid_tbl["num_further_refines"].value<unsigned int>();

    if (!grid_tbl["defect_pts"].is_array())
        throw std::invalid_argument("No defect_pts array in toml file");
    const auto defect_positions 
        = toml::convert<std::vector<std::vector<double>>>(*grid_tbl["defect_pts"].as_array());
    
    if (!grid_tbl["defect_refine_distances"].is_array())
        throw std::invalid_argument("No defect_refine_distances array in toml file");
    const auto defect_refine_distances
        = toml::convert<std::vector<double>>(*grid_tbl["defect_refine_distances"].as_array());

    const auto defect_radius = grid_tbl["defect_radius"].value<double>();
    const auto fix_defects = grid_tbl["fix_defects"].value<bool>();
    const auto grid_filename = grid_tbl["grid_filename"].value<std::string>();

    if (!pds_tbl["file_output"].is_table())
        throw std::invalid_argument("No perturbative_director_system.file_output table in toml file");
    const toml::table& file_tbl = *pds_tbl["file_output"].as_table();

    const auto data_folder = file_tbl["data_folder"].value<std::string>();
    const auto solution_vtu_filename = file_tbl["solution_vtu_filename"].value<std::string>();
    const auto rhs_vtu_filename = file_tbl["rhs_vtu_filename"].value<std::string>();
    const auto outer_structure_filename = file_tbl["outer_structure_filename"].value<std::string>(); 
    const auto inner_structure_filename = file_tbl["inner_structure_filename"].value<std::string>(); 
    const auto dataset_name = file_tbl["dataset_name"].value<std::string>(); 
    const auto core_structure_filename = file_tbl["core_structure_filename"].value<std::string>();  
    const auto pos_dataset_name = file_tbl["pos_dataset_name"].value<std::string>(); 
    const auto neg_dataset_name = file_tbl["neg_dataset_name"].value<std::string>(); 

    const auto refinement_level = file_tbl["refinement_level"].value<unsigned int>();
    const auto allow_merge = file_tbl["allow_merge"].value<bool>(); 
    auto max_boxes = file_tbl["max_boxes"].value<unsigned int>(); 

    if (!file_tbl["outer_point_set"].is_table())
        throw std::invalid_argument("No perturbative_director_system.file_output.outer_point_set table in toml file");
    const toml::table& outer_point_tbl = *file_tbl["outer_point_set"].as_table();

    const auto outer_center
        = toml::convert<std::vector<double>>(*outer_point_tbl["center"].as_array());
    const auto outer_r_0 = outer_point_tbl["r_0"].value<double>();
    const auto outer_r_f = outer_point_tbl["r_f"].value<double>();
    const auto outer_n_r = outer_point_tbl["n_r"].value<unsigned int>();
    const auto outer_n_theta = outer_point_tbl["n_theta"].value<unsigned int>();

    if (!file_tbl["inner_point_set"].is_table())
        throw std::invalid_argument("No perturbative_director_system.file_output.inner_point_set table in toml file");
    const toml::table& inner_point_tbl = *file_tbl["inner_point_set"].as_table();

    const auto inner_center
        = toml::convert<std::vector<double>>(*inner_point_tbl["center"].as_array());
    const auto inner_r_0 = inner_point_tbl["r_0"].value<double>();
    const auto inner_r_f = inner_point_tbl["r_f"].value<double>();
    const auto inner_n_r = inner_point_tbl["n_r"].value<unsigned int>();
    const auto inner_n_theta = inner_point_tbl["n_theta"].value<unsigned int>();

    if (!pds_tbl["boundary_conditions"].is_table())
        throw std::invalid_argument("No perturbative_director_system.boundary_conditions table in toml file");
    const auto boundary_condition_arg = pds_tbl["boundary_conditions"]["boundary_condition"].value<std::string>();

    // Check that all inputs are valid
    if (!degree) throw std::invalid_argument("No degree in toml file");
    if (!solver_type_arg) throw std::invalid_argument("No solver_type in toml file");
    if (!eps) throw std::invalid_argument("No eps in toml file");

    if (!left) throw std::invalid_argument("No left in toml file");
    if (!right) throw std::invalid_argument("No right in toml file");
    if (!grid_name) throw std::invalid_argument("No grid_name in toml file");
    if (!grid_parameters) throw std::invalid_argument("No grid_parameters in toml file");
    if (!num_refines) throw std::invalid_argument("No num_refines in toml file");
    if (!num_further_refines) throw std::invalid_argument("No num_further_refines in toml file");

    if (!defect_radius) throw std::invalid_argument("No defect_radius in toml file");
    if (!fix_defects) throw std::invalid_argument("No fix_defects in toml file");
    if (!grid_filename) throw std::invalid_argument("No grid_filename in toml file");

    if (!data_folder) throw std::invalid_argument("No data_folder in toml file");
    if (!solution_vtu_filename) throw std::invalid_argument("No solution_vtu_filename in toml file");
    if (!rhs_vtu_filename) throw std::invalid_argument("No rhs_vtu_filename in toml file");
    if (!outer_structure_filename) throw std::invalid_argument("No outer_structure_filename in toml file");
    if (!inner_structure_filename) throw std::invalid_argument("No inner_structure_filename in toml file");
    if (!dataset_name) throw std::invalid_argument("No dataset_name in toml file");
    if (!core_structure_filename) throw std::invalid_argument("No core_structure_filename in toml file");
    if (!pos_dataset_name) throw std::invalid_argument("No pos_dataset_name in toml file");
    if (!neg_dataset_name) throw std::invalid_argument("No neg_dataset_name in toml file");

    if (!refinement_level) throw std::invalid_argument("No refinement_level in toml file");
    if (!allow_merge) throw std::invalid_argument("No allow_merge in toml file");
    if (!max_boxes) throw std::invalid_argument("No max_boxes in toml file");

    if (!outer_r_0) throw std::invalid_argument("No outer r_0 in toml file");
    if (!outer_r_f) throw std::invalid_argument("No outer r_f in toml file");
    if (!outer_n_r) throw std::invalid_argument("No outer n_r in toml file");
    if (!outer_n_theta) throw std::invalid_argument("No outer n_theta in toml file");

    if (!inner_r_0) throw std::invalid_argument("No inner r_0 in toml file");
    if (!inner_r_f) throw std::invalid_argument("No inner r_f in toml file");
    if (!inner_n_r) throw std::invalid_argument("No inner n_r in toml file");
    if (!inner_n_theta) throw std::invalid_argument("No inner n_theta in toml file");

    if (!boundary_condition_arg) throw std::invalid_argument("No boundary_condition in toml file");

    // Do extra steps to get proper data types
    PerturbativeDirectorSystem<dim>::SolverType solver_type;
    if (solver_type_arg == "CG")
        solver_type = PerturbativeDirectorSystem<dim>::SolverType::CG;
    else if (solver_type_arg == "Direct")
        solver_type = PerturbativeDirectorSystem<dim>::SolverType::Direct;
    else
        throw std::invalid_argument("Inputted an incorrect solver type in toml file");

    GridTools::RadialPointSet<dim> outer_point_set;
    outer_point_set.center[0] = outer_center[0];
    outer_point_set.center[1] = outer_center[1];
    outer_point_set.r_0 = outer_r_0.value();
    outer_point_set.r_f = outer_r_f.value();
    outer_point_set.n_r = outer_n_r.value();
    outer_point_set.n_theta = outer_n_theta.value();
    
    GridTools::RadialPointSet<dim> inner_point_set;
    inner_point_set.center[0] = inner_center[0];
    inner_point_set.center[1] = inner_center[1];
    inner_point_set.r_0 = inner_r_0.value();
    inner_point_set.r_f = inner_r_f.value();
    inner_point_set.n_r = inner_n_r.value();
    inner_point_set.n_theta = inner_n_theta.value();

    std::vector<dealii::Point<dim>> defect_pts(defect_positions.size());
    for (std::size_t i = 0; i < defect_pts.size(); ++i)
        for (unsigned int j = 0; j < dim; ++j)
            defect_pts[i][j] = defect_positions[i][j];

    if (max_boxes.value() == 0)
        max_boxes.value() = dealii::numbers::invalid_unsigned_int;


    PerturbativeDirectorSystem<dim>::BoundaryCondition boundary_condition;
    if (boundary_condition_arg.value() == "Neumann")
        boundary_condition = PerturbativeDirectorSystem<dim>::BoundaryCondition::Neumann;
    else if (boundary_condition_arg.value() == "Dirichlet")
        boundary_condition = PerturbativeDirectorSystem<dim>::BoundaryCondition::Dirichlet;
    else
        throw std::invalid_argument("Inputted incorrect boundary condition in toml file");

    std::unique_ptr<dealii::Function<dim>> 
        righthand_side = std::make_unique<PerturbativeDirectorRighthandSide<dim>>(defect_charges,
                                                                                  defect_pts);
    std::unique_ptr<dealii::Function<dim>> 
        boundary_function = std::make_unique<PerturbativeDirectorBoundaryCondition<dim>>(defect_charges,
                                                                                         defect_pts,
                                                                                         eps.value());

    try
    {
        PerturbativeDirectorSystem<dim> perturbative_director_system(degree.value(),
                                                                     grid_name.value(),
                                                                     grid_parameters.value(),
                                                                     left.value(),
                                                                     right.value(),
                                                                     num_refines.value(),
                                                                     num_further_refines.value(),
                                                                     defect_pts,
                                                                     defect_refine_distances,
                                                                     defect_radius.value(),
                                                                     fix_defects.value(),
                                                                     grid_filename.value(),

                                                                     solver_type,

                                                                     data_folder.value(),
                                                                     solution_vtu_filename.value(),
                                                                     rhs_vtu_filename.value(),
                                                                     outer_structure_filename.value(),
                                                                     inner_structure_filename.value(),
                                                                     dataset_name.value(),
                                                                     core_structure_filename.value(),
                                                                     pos_dataset_name.value(),
                                                                     neg_dataset_name.value(),

                                                                     outer_point_set,
                                                                     inner_point_set,
                                                                     refinement_level.value(),
                                                                     allow_merge.value(),
                                                                     max_boxes.value(),
                                                                     boundary_condition,
                                                                     std::move(righthand_side),
                                                                     std::move(boundary_function));
        perturbative_director_system.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
 
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
 
    return 0;
}
