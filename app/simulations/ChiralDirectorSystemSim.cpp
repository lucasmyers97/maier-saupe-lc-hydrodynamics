#include "LiquidCrystalSystems/ChiralDirectorSystem.cpp"
#include "LiquidCrystalSystems/ChiralDirectorSystem.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include "Parameters/toml.hpp"

#include <memory>
#include <string>

constexpr int dim = 2;

std::unique_ptr<ChiralDirectorSystem<dim>> 
parse_parameter_file(const toml::table& tbl) 
{
    // get all values from toml file
    const auto zeta = tbl["system"]["zeta"].value<double>();
    const auto alpha = tbl["system"]["alpha"].value<double>();
    const auto d = tbl["system"]["d"].value<double>();

    const auto grid_name = tbl["grid"]["name"].value<std::string>();

    if (!tbl["grid"]["center"].is_array()) 
        throw std::invalid_argument("No center array in toml file");
    const auto grid_center
        = toml::convert<std::vector<double>>(*tbl["grid"]["center"].as_array());

    const auto grid_radius = tbl["grid"]["radius"].value<double>();
    const auto num_refines = tbl["grid"]["num_refines"].value<unsigned int>();
    const auto num_further_refines = tbl["grid"]["num_further_refines"].value<unsigned int>();

    if (!tbl["grid"]["defect_pts"].is_array())
        throw std::invalid_argument("No defect_pts array in toml file");
    const auto defect_pts
        = toml::convert<std::vector<std::vector<double>>>(*tbl["grid"]["defect_pts"].as_array());

    if (!tbl["grid"]["defect_refine_distances"].is_array()) 
        throw std::invalid_argument("No defect_refine_distances array in toml file");
    const auto defect_refine_distances
        = toml::convert<std::vector<double>>(*tbl["grid"]["defect_refine_distances"].as_array());

    const auto defect_radius = tbl["grid"]["defect_radius"].value<double>();
    const auto mesh_filename = tbl["grid"]["mesh_filename"].value<std::string>();

    const auto degree = tbl["finite_element"]["degree"].value<unsigned int>();
    const auto solver_type = tbl["finite_element"]["solver_type"].value<std::string>();

    const auto data_folder = tbl["output"]["data_folder"].value<std::string>();    
    const auto solution_vtu_filename = tbl["output"]["solution_vtu_filename"].value<std::string>();
    const auto rhs_vtu_filename = tbl["output"]["rhs_vtu_filename"].value<std::string>();

    const auto outer_structure_filename = tbl["output"]["h5"]["outer_structure_filename"].value<std::string>();
    const auto dataset_name = tbl["output"]["h5"]["dataset_name"].value<std::string>();

    const auto core_structure_filename = tbl["output"]["h5"]["core_structure_filename"].value<std::string>();
    const auto pos_dataset_name = tbl["output"]["h5"]["pos_dataset_name"].value<std::string>();
    const auto neg_dataset_name = tbl["output"]["h5"]["neg_dataset_name"].value<std::string>();

    const auto refinement_level = tbl["output"]["h5"]["refinement_level"].value<unsigned int>();
    const auto allow_merge = tbl["output"]["h5"]["allow_merge"].value<bool>();
    const auto max_boxes = tbl["output"]["h5"]["max_boxes"].value<int>();
    
    if (!tbl["output"]["h5"]["point_set"]["center"].is_array()) 
        throw std::invalid_argument("No point_set center array in toml file");
    const auto point_set_center
        = toml::convert<std::vector<double>>(*tbl["output"]["h5"]["point_set"]["center"].as_array());

    const auto r_0 = tbl["output"]["h5"]["point_set"]["r_0"].value<double>();
    const auto r_f = tbl["output"]["h5"]["point_set"]["r_f"].value<double>();
    const auto n_r = tbl["output"]["h5"]["point_set"]["n_r"].value<unsigned int>();
    const auto n_theta = tbl["output"]["h5"]["point_set"]["n_theta"].value<unsigned int>();

    // check that everything is parsed properly
    if (!zeta) throw std::invalid_argument("No zeta in toml file");
    if (!alpha) throw std::invalid_argument("No alpha in toml file");
    if (!d) throw std::invalid_argument("No d in toml file");

    if (!grid_name) throw std::invalid_argument("No name in toml file");

    if (!grid_radius) throw std::invalid_argument("No grid radius in toml file");
    if (!num_refines) throw std::invalid_argument("No num_refines in toml file");
    if (!num_further_refines) throw std::invalid_argument("No num_further_refines in toml file");

    if (!defect_radius) throw std::invalid_argument("No defect_radius in toml file");
    if (!mesh_filename) throw std::invalid_argument("No mesh_filename in toml file");

    if (!degree) throw std::invalid_argument("No degree in toml file");
    if (!solver_type) throw std::invalid_argument("No solver_type in toml file");

    if (!data_folder) throw std::invalid_argument("No data_folder in toml file");    
    if (!solution_vtu_filename) throw std::invalid_argument("No solution_vtu_filename in toml file");
    if (!rhs_vtu_filename) throw std::invalid_argument("No rhs_vtu_filename in toml file");

    if (!outer_structure_filename) throw std::invalid_argument("No outer_structure_filename in toml file");
    if (!dataset_name) throw std::invalid_argument("No dataset_name in toml file");

    if (!core_structure_filename) throw std::invalid_argument("No core_structure_filename in toml file");
    if (!pos_dataset_name) throw std::invalid_argument("No pos_dataset_name in toml file");
    if (!neg_dataset_name) throw std::invalid_argument("No neg_dataset_name in toml file");

    if (!refinement_level) throw std::invalid_argument("No refinement_level in toml file");
    if (!allow_merge) throw std::invalid_argument("No allow_merge in toml file");
    if (!max_boxes) throw std::invalid_argument("No max_boxes in toml file");

    if (!r_0) throw std::invalid_argument("No r_0 in toml file");
    if (!r_f) throw std::invalid_argument("No r_f in toml file");
    if (!n_r) throw std::invalid_argument("No n_r in toml file");
    if (!n_theta) throw std::invalid_argument("No n_theta in toml file");

    GridTools::RadialPointSet<dim> point_set;
    point_set.center[0] = point_set_center[0];
    point_set.center[1] = point_set_center[1];
    point_set.r_0 = r_0.value();
    point_set.r_f = r_f.value();
    point_set.n_r = n_r.value();
    point_set.n_theta = n_theta.value();

    ChiralDirectorSystem<dim>::SolverType solver_type_obj;
    if (solver_type.value() == "CG")
        solver_type_obj = ChiralDirectorSystem<dim>::SolverType::CG;
    else if (solver_type.value() == "Direct")
        solver_type_obj = ChiralDirectorSystem<dim>::SolverType::Direct;
    else
        throw std::invalid_argument("Invalid option for solver_type in toml");

    // get boundary functions
    auto righthand_side = std::make_unique<ChiralDirectorRighthandSide<dim>>(d.value());
    auto boundary_function = std::make_unique<ChiralDirectorBoundaryCondition<dim>>(d.value());

    dealii::Point<dim> grid_center_point(grid_center[0], grid_center[1]);
    std::vector<dealii::Point<dim>> defect_pts_point(defect_pts.size());
    for (std::size_t i = 0; i < defect_pts.size(); ++i)
        defect_pts_point[i] = dealii::Point<dim>(defect_pts[i][0], defect_pts[i][1]);

    auto chiral_director_system 
        = std::make_unique<ChiralDirectorSystem<dim>> (degree.value(),

                                                       alpha.value(),
                                                       zeta.value(),

                                                       grid_name.value(),
                                                       grid_center_point,
                                                       grid_radius.value(),
                                                       num_refines.value(),
                                                       num_further_refines.value(),
                                                       defect_pts_point,
                                                       defect_refine_distances,
                                                       defect_radius.value(),
                                                       mesh_filename.value(),

                                                       solver_type_obj,

                                                       data_folder.value(),
                                                       solution_vtu_filename.value(),
                                                       rhs_vtu_filename.value(),
                                                       outer_structure_filename.value(),
                                                       dataset_name.value(),
                                                       core_structure_filename.value(),
                                                       pos_dataset_name.value(),
                                                       neg_dataset_name.value(),

                                                       point_set,
                                                       refinement_level.value(),
                                                       allow_merge.value(),
                                                       max_boxes.value(),
                                                       std::move(righthand_side),
                                                       std::move(boundary_function));

    return chiral_director_system;
}

int main(int ac, char *av[])
{
    try
    {
        if (ac - 1 != 1)
            throw std::invalid_argument("Error! Didn't input filename");
        std::string toml_filename(av[1]);

        const toml::table tbl = toml::parse_file(toml_filename);

        unsigned int degree = 1;

        const double zeta = 0.0;
        const double alpha = 0.0;
        const double d = 0.5;

        // grid parameters
        std::string grid_name = "hyper_ball_balanced";
        dealii::Point<dim> grid_center = {0, 0};
        double grid_radius = 1.0;
        unsigned int num_refines = 3;
        unsigned int num_further_refines = 2;
        std::vector<dealii::Point<dim>> defect_pts(2);
        defect_pts[0] = dealii::Point<dim>({-0.25, 0.0});
        defect_pts[1] = dealii::Point<dim>({0.25, 0.0});
        std::vector<double> defect_refine_distances = {0.25};
        double defect_radius = 0.25;

        // std::string grid_filename = "/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/jonas-grid/circle_grid.msh";
        std::string grid_filename = "";

        ChiralDirectorSystem<dim>::SolverType solver_type 
            = ChiralDirectorSystem<dim>::SolverType::CG;

        // output parameters
        std::string data_folder = "/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/chiral-director/";
        std::string solution_vtu_filename = "theta_c_solution";
        std::string rhs_vtu_filename = "system_rhs";

        std::string outer_structure_filename = "outer_structure.h5";
        std::string dataset_name = "director_perturbation";

        std::string core_structure_filename = "core_structure.h5";
        std::string pos_dataset_name = "pos_phi";
        std::string neg_dataset_name = "neg_phi";

        GridTools::RadialPointSet<dim> point_set;
        point_set.center = dealii::Point<dim>({0.0, 0.0});
        point_set.r_0 = 100;
        point_set.r_f = 5400;
        point_set.n_r = 2000;
        point_set.n_theta = 1000;

        unsigned int refinement_level = 3;
        bool allow_merge = false;
        unsigned int max_boxes = dealii::numbers::invalid_unsigned_int;

        std::unique_ptr<dealii::Function<dim>> 
            righthand_side = std::make_unique<ChiralDirectorRighthandSide<dim>>(d);
        std::unique_ptr<dealii::Function<dim>> 
        boundary_function = std::make_unique<ChiralDirectorBoundaryCondition<dim>>(d);

        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

        ChiralDirectorSystem<dim> chiral_director_system(degree,

                                                         alpha,
                                                         zeta,

                                                         grid_name,
                                                         grid_center,
                                                         grid_radius,
                                                         num_refines,
                                                         num_further_refines,
                                                         defect_pts,
                                                         defect_refine_distances,
                                                         defect_radius,
                                                         grid_filename,

                                                         solver_type,

                                                         data_folder,
                                                         solution_vtu_filename,
                                                         rhs_vtu_filename,
                                                         outer_structure_filename,
                                                         dataset_name,
                                                         core_structure_filename,
                                                         pos_dataset_name,
                                                         neg_dataset_name,

                                                         point_set,
                                                         refinement_level,
                                                         allow_merge,
                                                         max_boxes,
                                                         std::move(righthand_side),
                                                         std::move(boundary_function));
        chiral_director_system.run();
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
