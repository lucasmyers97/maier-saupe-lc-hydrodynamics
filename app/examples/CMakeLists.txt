# Add all of the examples executables, link dependencies
add_executable(atan_test atan_test.cpp)

add_executable(bit_test bit_test.cpp)

add_executable(dealii_function_test dealii_function_test.cpp)
target_link_libraries(dealii_function_test PRIVATE deal.II::Interface)

# add_executable(lagrange_test lagrange_test.cpp)
# target_link_libraries(lagrange_test MaierSaupe)

# add_executable(lagrange_time_test lagrange_time_test.cpp)
# target_link_libraries(lagrange_time_test MaierSaupe deal.II::Interface)

add_executable(lapack_matrix_test lapack_matrix_test.cpp)
target_link_libraries(lapack_matrix_test deal.II::Interface)

add_executable(lebedev_pointer_test lebedev_pointer_test.cpp)
target_link_libraries(lebedev_pointer_test SphereLebedevRule)

add_executable(lu_matrix_test lu_matrix_test.cpp)
target_link_libraries(lu_matrix_test MaierSaupe)

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
  deal.II::Interface HighFive BoundaryValues)

add_executable(serialize_sparsity_pattern serialize_sparsity_pattern.cpp)
target_link_libraries(serialize_sparsity_pattern
  deal.II::Interface Boost::serialization)

add_executable(high_five_example high_five_example.cpp)
target_link_libraries(high_five_example
  Boost::program_options HighFive)

add_executable(serialize_triangulation serialize_triangulation.cpp)
target_link_libraries(serialize_triangulation
  deal.II::Interface Boost::serialization)

add_executable(serialize_dof_handler serialize_dof_handler.cpp)
target_link_libraries(serialize_dof_handler
  deal.II::Interface Boost::serialization)

add_executable(linear_interpolation_example linear_interpolation_example.cpp)
target_link_libraries(linear_interpolation_example
  LinearInterpolation HighFive deal.II::Interface)

install(TARGETS dealii_function_test atan_test bit_test 
        # lagrange_test lagrange_time_test 
        lapack_matrix_test lebedev_pointer_test
        lu_matrix_test max_element plot_gaussian plot_uniaxial_nematic
        poisson_test plot_defect_configuration  serialize_sparsity_pattern
        high_five_example serialize_triangulation serialize_dof_handler
        linear_interpolation_example
        DESTINATION bin/examples)
