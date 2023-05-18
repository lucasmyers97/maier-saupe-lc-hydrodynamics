#include "LiquidCrystalSystems/PerturbativeDirectorSystem.cpp"

#include <deal.II/base/mpi.h>

#include <memory>

int main(int argc, char *argv[])
{
    constexpr int dim = 2;

    unsigned int degree = 2;

    // grid parameters
    double left = -1064.0;
    double right = 1064.0;
    unsigned int num_refines = 8;
    unsigned int num_further_refines = 3;
    std::vector<dealii::Point<dim>> defect_pts(2);
    defect_pts[0] = dealii::Point<dim>({-30.0, 0.0});
    defect_pts[1] = dealii::Point<dim>({30.0, 0.0});
    // std::vector<double> defect_refine_distances = {10.0, 20.0, 30.0};
    std::vector<double> defect_refine_distances = {2.0, 5.0, 10.0, 20.0, 30.0};
    double defect_radius = 2.5;
    bool fix_defects = false;

    // output parameters
    std::string h5_filename = "./temp-data/carter-numerical-solution/outer_structure.h5";
    std::string dataset_name = "director_perturbation";
    GridTools::RadialPointSet<dim> point_set;
    point_set.center = dealii::Point<dim>({0.0, 0.0});
    point_set.r_0 = 100;
    point_set.r_f = 1000;
    point_set.n_r = 2000;
    point_set.n_theta = 1000;

    unsigned int refinement_level = 3;
    bool allow_merge = false;
    unsigned int max_boxes = dealii::numbers::invalid_unsigned_int;

    PerturbativeDirectorSystem<dim>::BoundaryCondition 
        boundary_condition = PerturbativeDirectorSystem<dim>::BoundaryCondition::Neumann;

    std::vector<double> defect_charges = {0.5, -0.5};
    double eps = 0.1;

    std::unique_ptr<PerturbativeDirectorRighthandSide<dim>> 
        righthand_side = std::make_unique<PerturbativeDirectorRighthandSide<dim>>(defect_charges,
                                                                                  defect_pts,
                                                                                  eps);

    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        PerturbativeDirectorSystem<dim> perturbative_director_system(degree,
                                                                     left,
                                                                     right,
                                                                     num_refines,
                                                                     num_further_refines,
                                                                     defect_pts,
                                                                     defect_refine_distances,
                                                                     defect_radius,
                                                                     fix_defects,
                                                                     h5_filename,
                                                                     dataset_name,
                                                                     point_set,
                                                                     refinement_level,
                                                                     allow_merge,
                                                                     max_boxes,
                                                                     boundary_condition,
                                                                     std::move(righthand_side));
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
