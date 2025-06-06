find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)
find_package(MPI REQUIRED)

# Add all of the examples executables, link dependencies
add_executable(atan_test atan_test.cpp)

add_executable(bit_test bit_test.cpp)

add_executable(dealii_function_test dealii_function_test.cpp)
target_link_libraries(dealii_function_test deal.II::Interface)

add_executable(lapack_matrix_test lapack_matrix_test.cpp)
target_link_libraries(lapack_matrix_test deal.II::Interface)

add_executable(lebedev_pointer_test lebedev_pointer_test.cpp)
target_link_libraries(lebedev_pointer_test SphereLebedevRule)

add_executable(lu_matrix_test lu_matrix_test.cpp)
target_link_libraries(lu_matrix_test LUMatrix)

add_executable(max_element max_element.cpp)
target_link_libraries(max_element deal.II::Interface)

add_executable(plot_gaussian plot_gaussian.cpp)
target_link_libraries(plot_gaussian deal.II::Interface)

add_executable(plot_uniaxial_nematic plot_uniaxial_nematic.cpp)
target_link_libraries(plot_uniaxial_nematic deal.II::Interface BoundaryValues)

add_executable(poisson_test poisson_test.cpp)
target_link_libraries(poisson_test deal.II::Interface)

add_executable(plot_defect_configuration plot_defect_configuration.cpp)
target_link_libraries(plot_defect_configuration
  deal.II::Interface
  HighFive
  BoundaryValues
  MaierSaupeConstants)

add_executable(serialize_sparsity_pattern serialize_sparsity_pattern.cpp)
target_link_libraries(serialize_sparsity_pattern
  deal.II::Interface
  Boost::serialization)

add_executable(high_five_example high_five_example.cpp)
target_link_libraries(high_five_example
  Boost::program_options
  HighFive)

add_executable(serialize_triangulation serialize_triangulation.cpp)
target_link_libraries(serialize_triangulation
  deal.II::Interface
  Boost::serialization)

add_executable(serialize_dof_handler serialize_dof_handler.cpp)
target_link_libraries(serialize_dof_handler
  deal.II::Interface
  Boost::serialization)

add_executable(linear_interpolation_example linear_interpolation_example.cpp)
target_link_libraries(linear_interpolation_example
  LinearInterpolation
  HighFive
  deal.II::Interface)

add_executable(project_seg_fault project_seg_fault.cpp)
target_link_libraries(project_seg_fault
                      deal.II::Interface)

add_executable(minimal_distributed_bug minimal_distributed_bug.cpp)
target_link_libraries(minimal_distributed_bug
                      deal.II::Interface)

add_executable(step-40 step-40.cc)
target_link_libraries(step-40
  PRIVATE
  deal.II::Interface)

add_executable(active_flow_solver active_flow_solver.cpp)
target_link_libraries(active_flow_solver
  PRIVATE
  deal.II::Interface)

add_executable(dealii_subdivided_grid dealii_subdivided_grid.cpp)
target_link_libraries(dealii_subdivided_grid
  PRIVATE
  deal.II::Interface)

add_executable(program_options_vector program_options_vector.cpp)
target_link_libraries(program_options_vector
  PRIVATE
  Boost::program_options)


add_executable(check_diagonalization_schemes check_diagonalization_schemes.cpp)
target_link_libraries(check_diagonalization_schemes
  PRIVATE
  deal.II::Interface
  LagrangeMultiplier)


add_executable(lagrange_test lagrange_test.cpp)
target_link_libraries(lagrange_test
  PRIVATE
  deal.II::Interface
  LagrangeMultiplier)

add_executable(reduced_lagrange_test reduced_lagrange_test.cpp)
target_link_libraries(reduced_lagrange_test
  PRIVATE
  deal.II::Interface
  LagrangeMultiplier
  LagrangeMultiplierReduced)


add_executable(zumer_domain zumer_domain.cpp)
target_link_libraries(zumer_domain
  PRIVATE
  deal.II::Interface)

add_executable(sqrt_test sqrt_test.cpp)

add_executable(lagrange_continuity_check lagrange_continuity_check.cpp)
target_link_libraries(lagrange_continuity_check
  PRIVATE
  LagrangeMultiplier
  deal.II::Interface
  HighFive)


add_executable(analytic_degenerate_eigenvalue analytic_degenerate_eigenvalue.cpp)
target_link_libraries(analytic_degenerate_eigenvalue
  PRIVATE
  deal.II::Interface
  MaierSaupeConstants
  LagrangeMultiplierReduced
  LagrangeMultiplier)


add_executable(lagrange_multiplier_analytic_test lagrange_multiplier_analytic_test.cpp)
target_link_libraries(lagrange_multiplier_analytic_test
  PRIVATE
  deal.II::Interface
  LagrangeMultiplierAnalytic
  LagrangeMultiplier
  MaierSaupeConstants)

add_executable(tensor_to_vector_test tensor_to_vector_test.cpp)
target_link_libraries(tensor_to_vector_test
  PRIVATE
  ExampleFunctions
  deal.II::Interface)

add_executable(project_two_defect project_two_defect.cpp)
target_link_libraries(project_two_defect
  PRIVATE
  deal.II::Interface
  MaierSaupeConstants
  ExampleFunctions
  Postprocessors)

add_executable(serialize_boundary_values serialize_boundary_values.cpp)
target_link_libraries(serialize_boundary_values
  PRIVATE
  Boost::serialization
  BoundaryValues)

add_executable(boundary_values_parameter_pack boundary_values_parameter_pack.cpp)
target_link_libraries(boundary_values_parameter_pack
  PRIVATE
  BoundaryValues
  Boost::headers)

add_executable(check_set_order check_set_order.cpp)
target_link_libraries(check_set_order
  PRIVATE
  LagrangeMultiplier
  LagrangeMultiplierReduced
  deal.II::Interface
  MaierSaupeConstants)

add_executable(serialize_lagrange_reduced serialize_lagrange_reduced.cpp)
target_link_libraries(serialize_lagrange_reduced
  PRIVATE
  LagrangeMultiplierReduced
  Boost::serialization)

add_executable(serialize_lagrange_analytic serialize_lagrange_analytic.cpp)
target_link_libraries(serialize_lagrange_analytic
  PRIVATE
  LagrangeMultiplierAnalytic
  Boost::serialization)

add_executable(serialize_boundary_pointer serialize_boundary_pointer.cpp)
target_link_libraries(serialize_boundary_pointer
  PRIVATE
  BoundaryValues
  Boost::headers
  Boost::serialization)

# add_executable(deserialize_nematic_mpi deserialize_nematic_mpi.cpp)
# target_link_libraries(deserialize_nematic_mpi
#   PRIVATE
#   BasicDrivers
#   deal.II::Interface)

add_executable(print_nematic_hydro_parameters print_nematic_hydro_parameters.cpp)
target_link_libraries(print_nematic_hydro_parameters
  PRIVATE
  BasicDrivers
  deal.II::Interface)

add_executable(print_hydro_parameters print_hydro_parameters.cpp)
target_link_libraries(print_hydro_parameters
  PRIVATE
  LiquidCrystalSystems
  deal.II::Interface)

add_executable(print_nematic_parameters print_nematic_parameters.cpp)
target_link_libraries(print_nematic_parameters
  PRIVATE
  BasicDrivers
  deal.II::Interface)

add_executable(linspace_test linspace_test.cpp)
target_link_libraries(linspace_test
  PRIVATE
  NumericalTools)

add_executable(output_Q_values output_Q_values.cpp)
target_link_libraries(output_Q_values
  PRIVATE
  BasicDrivers
  LiquidCrystalSystems
  Serialization
  Postprocessors
  deal.II::Interface)

add_executable(lagrange_copy_test lagrange_copy_test.cpp)
target_link_libraries(lagrange_copy_test
  PRIVATE
  LagrangeMultiplierAnalytic
  deal.II::Interface)

add_executable(tuple_test tuple_test.cpp)

add_executable(calculate_defect_quantities calculate_defect_quantities.cpp)
target_link_libraries(calculate_defect_quantities
    PRIVATE
    deal.II::Interface
    LiquidCrystalSystems
    Serialization
    NumericalTools
    HighFive)

add_executable(find_local_minima find_local_minima.cpp)
target_link_libraries(find_local_minima
    PRIVATE
    deal.II::Interface
    LiquidCrystalSystems
    Serialization
    NumericalTools
    HighFive)

add_executable(output_archived_nematic_config output_archived_nematic_config.cpp)
target_link_libraries(output_archived_nematic_config
    PRIVATE
    deal.II::Interface
    LiquidCrystalSystems
    Serialization)

add_executable(dealii_mpi_allgather dealii_mpi_allgather.cpp)
target_link_libraries(dealii_mpi_allgather
    PRIVATE
    deal.II::Interface)

add_executable(dealii_hdf5_write dealii_hdf5_write.cpp)
target_link_libraries(dealii_hdf5_write
    PRIVATE
    deal.II::Interface)

add_executable(check_anisotropic_elasticity check_anisotropic_elasticity.cpp)
target_link_libraries(check_anisotropic_elasticity
    PRIVATE
    deal.II::Interface
    LiquidCrystalSystems)

add_executable(distributed_vector_hdf5_example distributed_vector_hdf5_example.cpp)
target_link_libraries(distributed_vector_hdf5_example
    PRIVATE
    deal.II::Interface
    Output)

add_executable(defect_grid_generator defect_grid_generator.cpp)
target_link_libraries(defect_grid_generator
    PRIVATE
    deal.II::Interface
    DefectGridGenerator)

add_executable(dzyaloshinskii_function dzyaloshinskii_function.cpp)
target_link_libraries(dzyaloshinskii_function
    PRIVATE
    deal.II::Interface
    BoundaryValues)

add_executable(mesh_integral_test mesh_integral_test.cpp)
target_link_libraries(mesh_integral_test
    PRIVATE
    deal.II::Interface)

add_executable(check_defect_domains check_defect_domains.cpp)
target_link_libraries(check_defect_domains
    PRIVATE
    SetDefectBoundaryConstraints)

add_executable(parameter_regex parameter_regex.cpp)

add_executable(grid_level_test grid_level_test.cpp)
target_link_libraries(grid_level_test
    PRIVATE
    deal.II::Interface)

add_executable(test_compute_set_union test_compute_set_union.cpp)
target_link_libraries(test_compute_set_union
    PRIVATE
    deal.II::Interface)

add_executable(test_parameter_parser test_parameter_parser.cpp)
target_link_libraries(test_parameter_parser
    PRIVATE
    ParameterParser
    deal.II::Interface)

add_executable(mpi_test mpi_test.cpp)
target_link_libraries(mpi_test
    PRIVATE
    MPI::MPI_CXX)

add_executable(test_grid_cache_and_bounding_boxes 
               test_grid_cache_and_bounding_boxes.cpp)
target_link_libraries(test_grid_cache_and_bounding_boxes
    PRIVATE
    deal.II::Interface
    BasicDrivers
    LiquidCrystalSystems)

add_executable(test_multi_defect_configuration 
               test_multi_defect_configuration.cpp)
target_link_libraries(test_multi_defect_configuration
    PRIVATE
    deal.II::Interface
    BoundaryValues
    MaierSaupeConstants
    Postprocessors)

add_executable(test_internal_boundary_values test_internal_boundary_values.cpp)
target_link_libraries(test_internal_boundary_values
    PRIVATE
    deal.II::Interface
    BoundaryValues
    MaierSaupeConstants
    Postprocessors
    SetDefectBoundaryConstraints)

add_executable(parse_toml parse_toml.cpp)
target_link_libraries(parse_toml
    PRIVATE
    toml)

install(TARGETS 
  dealii_function_test 
  atan_test bit_test 
  lapack_matrix_test
  lebedev_pointer_test 
  lu_matrix_test 
  max_element 
  plot_gaussian
  plot_uniaxial_nematic 
  poisson_test 
  plot_defect_configuration
  serialize_sparsity_pattern 
  high_five_example serialize_triangulation
  serialize_dof_handler 
  linear_interpolation_example 
  project_seg_fault step-40
  minimal_distributed_bug
  active_flow_solver
  dealii_subdivided_grid
  program_options_vector
  check_diagonalization_schemes
  sqrt_test
  reduced_lagrange_test
  zumer_domain
  lagrange_test
  lagrange_continuity_check
  analytic_degenerate_eigenvalue
  lagrange_multiplier_analytic_test
  tensor_to_vector_test
  project_two_defect
  serialize_boundary_values
  boundary_values_parameter_pack
  check_set_order
  serialize_lagrange_reduced
  serialize_lagrange_analytic
  serialize_boundary_pointer
  # deserialize_nematic_mpi
  print_nematic_hydro_parameters
  print_hydro_parameters
  print_nematic_parameters
  output_Q_values
  linspace_test
  lagrange_copy_test
  tuple_test
  calculate_defect_quantities
  find_local_minima
  output_archived_nematic_config
  dealii_mpi_allgather
  dealii_hdf5_write
  check_anisotropic_elasticity
  distributed_vector_hdf5_example
  defect_grid_generator
  dzyaloshinskii_function
  mesh_integral_test
  check_defect_domains
  parameter_regex
  grid_level_test
  test_compute_set_union
  test_parameter_parser
  mpi_test
  test_grid_cache_and_bounding_boxes
  test_multi_defect_configuration
  test_internal_boundary_values
  parse_toml
  DESTINATION bin/examples)

# These things involve auto-diff which involves some compiler bug
# add_executable(simple_auto_diff simple_auto_diff.cpp)
# target_link_libraries(simple_auto_diff
#   PRIVATE
#   deal.II::Interface)
# 
# add_executable(rotation_auto_diff rotation_auto_diff.cpp)
# target_link_libraries(rotation_auto_diff
#   PRIVATE
#   deal.II::Interface
#   LagrangeMultiplier)
# 
# add_executable(ad_rotation_matrix ad_rotation_matrix.cpp)
# target_link_libraries(ad_rotation_matrix
#   PRIVATE
#   deal.II::Interface
#   MaierSaupeConstants
#   LagrangeMultiplierReduced
#   LagrangeMultiplier
#   NumericalTools)
# 
# add_executable(rotation_diff_identity rotation_diff_identity.cpp)
# target_link_libraries(rotation_diff_identity
#   PRIVATE
#   deal.II::Interface
#   LagrangeMultiplier)
# 
# add_executable(check_rotations check_rotations.cpp)
# target_link_libraries(check_rotations
#   PRIVATE
#   deal.II::Interface)
# 
# add_executable(analytic_eigenvector_derivative analytic_eigenvector_derivative.cpp)
# target_link_libraries(analytic_eigenvector_derivative
#   PRIVATE
#   deal.II::Interface)
# 
# add_executable(efficient_lagrange_multiplier efficient_lagrange_multiplier.cpp)
# target_link_libraries(efficient_lagrange_multiplier
#   PRIVATE
#   deal.II::Interface
#   MaierSaupeConstants
#   LagrangeMultiplier
#   LagrangeMultiplierEfficient)
# 
# add_executable(minimal_autodiff_bug minimal_autodiff_bug.cpp)
# target_link_libraries(minimal_autodiff_bug
#   PRIVATE
#   deal.II::Interface)
#
# add_executable(analytic_reduced_lagrange analytic_reduced_lagrange.cpp)
# target_link_libraries(analytic_reduced_lagrange
#   PRIVATE
#   deal.II::Interface
#   MaierSaupeConstants
#   LagrangeMultiplierReduced
#   LagrangeMultiplier)
# 
# install(TARGETS 
#   simple_auto_diff
#   rotation_auto_diff
#   rotation_diff_identity
#   check_rotations
#   efficient_lagrange_multiplier
#   minimal_autodiff_bug
#   analytic_eigenvector_derivative
#   ad_rotation_matrix
#   analytic_reduced_lagrange
#   DESTINATION bin/examples)
