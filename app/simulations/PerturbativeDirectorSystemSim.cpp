#include "LiquidCrystalSystems/PerturbativeDirectorSystem.cpp"
#include "LiquidCrystalSystems/PerturbativeDirectorSystem.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <memory>

int main(int argc, char *argv[])
{
    constexpr int dim = 2;

    unsigned int degree = 1;

    const double eps = 0.1;

    // grid parameters
    double left = -1064.0;
    double right = 1064.0;
    // std::string grid_name = "plate_with_a_hole";
    std::string grid_name = "hyper_ball_balanced";
    // std::string grid_parameters = "100 : 300 : 764 : 764 : 764 : 746 "
    //                               ": 0.0, 0.0 : 0 : 1 : 1 : 2 : false";
    // std::string grid_parameters = "300 : 600 : 464 : 464 : 464 : 446 "
    //                               ": 0.0, 0.0 : 0 : 1 : 1 : 2 : false";
    std::string grid_parameters = "0.0, 0.0 : 5500.0";
    unsigned int num_refines = 6;
    unsigned int num_further_refines = 3;
    // unsigned int num_refines = 2;
    // unsigned int num_further_refines = 0;
    // unsigned int num_refines = 5;
    // unsigned int num_further_refines = 0;
    std::vector<dealii::Point<dim>> defect_pts(2);
    defect_pts[0] = dealii::Point<dim>({-30.0, 0.0});
    defect_pts[1] = dealii::Point<dim>({30.0, 0.0});
    // defect_pts[0] = dealii::Point<dim>({1e-6, 0.0});
    // defect_pts[1] = dealii::Point<dim>({10000.0, 0.0});
    // std::vector<double> defect_refine_distances = {10.0, 20.0, 30.0};
    std::vector<double> defect_refine_distances = {50.0, 30.0, 20.0, 10.0, 5.0, 2.0};
    // std::vector<double> defect_refine_distances = {};
    double defect_radius = 10;
    bool fix_defects = true;

    // std::string grid_filename = "/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/jonas-grid/circle_grid.msh";
    std::string grid_filename = "";

    PerturbativeDirectorSystem<dim>::SolverType solver_type 
        = PerturbativeDirectorSystem<dim>::SolverType::CG;

    // output parameters
    std::string data_folder = "/home/lucas/Documents/research/maier-saupe-lc-hydrodynamics/temp-data/carter-numerical-solution/boundary-correct-code-archive/";
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

    PerturbativeDirectorSystem<dim>::BoundaryCondition 
        boundary_condition = PerturbativeDirectorSystem<dim>::BoundaryCondition::Neumann;

    std::vector<double> defect_charges = {0.5, -0.5};

    std::unique_ptr<dealii::Function<dim>> 
        righthand_side = std::make_unique<PerturbativeDirectorRighthandSide<dim>>(defect_charges,
                                                                                  defect_pts);
    std::unique_ptr<dealii::Function<dim>> 
        boundary_function = std::make_unique<PerturbativeDirectorBoundaryCondition<dim>>(defect_charges,
                                                                                         defect_pts,
                                                                                         eps);
    // std::unique_ptr<dealii::Function<dim>>
    //     boundary_function = std::make_unique<dealii::Functions::ZeroFunction<dim>>(2);

    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        PerturbativeDirectorSystem<dim> perturbative_director_system(degree,
                                                                     grid_name,
                                                                     grid_parameters,
                                                                     left,
                                                                     right,
                                                                     num_refines,
                                                                     num_further_refines,
                                                                     defect_pts,
                                                                     defect_refine_distances,
                                                                     defect_radius,
                                                                     fix_defects,
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
