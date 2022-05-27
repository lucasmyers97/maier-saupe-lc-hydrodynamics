find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)

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

add_executable(simple_auto_diff simple_auto_diff.cpp)
target_link_libraries(simple_auto_diff
  PRIVATE
  deal.II::Interface)

add_executable(rotation_auto_diff rotation_auto_diff.cpp)
target_link_libraries(rotation_auto_diff
  PRIVATE
  deal.II::Interface
  LagrangeMultiplier)

add_executable(check_diagonalization_schemes check_diagonalization_schemes.cpp)
target_link_libraries(check_diagonalization_schemes
  PRIVATE
  deal.II::Interface
  LagrangeMultiplier)

add_executable(rotation_diff_identity rotation_diff_identity.cpp)
target_link_libraries(rotation_diff_identity
  PRIVATE
  deal.II::Interface
  LagrangeMultiplier)

add_executable(check_rotations check_rotations.cpp)
target_link_libraries(check_rotations
  PRIVATE
  deal.II::Interface)

add_executable(reduced_lagrange_test reduced_lagrange_test.cpp)
target_link_libraries(reduced_lagrange_test
  PRIVATE
  deal.II::Interface
  LagrangeMultiplier
  LagrangeMultiplierReduced)

add_executable(efficient_lagrange_multiplier efficient_lagrange_multiplier.cpp)
target_link_libraries(efficient_lagrange_multiplier
  PRIVATE
  deal.II::Interface
  MaierSaupeConstants
  LagrangeMultiplier
  LagrangeMultiplierEfficient)

add_executable(minimal_autodiff_bug minimal_autodiff_bug.cpp)
target_link_libraries(minimal_autodiff_bug
  PRIVATE
  deal.II::Interface)

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

add_executable(analytic_eigenvector_derivative analytic_eigenvector_derivative.cpp)
target_link_libraries(analytic_eigenvector_derivative
  PRIVATE
  deal.II::Interface)

add_executable(analytic_reduced_lagrange analytic_reduced_lagrange.cpp)
target_link_libraries(analytic_reduced_lagrange
  PRIVATE
  deal.II::Interface
  MaierSaupeConstants
  LagrangeMultiplierReduced
  LagrangeMultiplier)

add_executable(ad_rotation_matrix ad_rotation_matrix.cpp)
target_link_libraries(ad_rotation_matrix
  PRIVATE
  deal.II::Interface
  MaierSaupeConstants
  LagrangeMultiplierReduced
  LagrangeMultiplier
  NumericalTools)

install(TARGETS dealii_function_test atan_test bit_test lapack_matrix_test
  lebedev_pointer_test lu_matrix_test max_element plot_gaussian
  plot_uniaxial_nematic poisson_test plot_defect_configuration
  serialize_sparsity_pattern high_five_example serialize_triangulation
  serialize_dof_handler linear_interpolation_example project_seg_fault step-40
  minimal_distributed_bug
  active_flow_solver
  dealii_subdivided_grid
  program_options_vector
  simple_auto_diff
  rotation_auto_diff
  check_diagonalization_schemes
  sqrt_test
  rotation_diff_identity
  check_rotations
  reduced_lagrange_test
  efficient_lagrange_multiplier
  minimal_autodiff_bug
  zumer_domain
  lagrange_continuity_check
  analytic_eigenvector_derivative
  analytic_reduced_lagrange
  ad_rotation_matrix
  DESTINATION bin/examples)
