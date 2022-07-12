find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)

add_executable(output_cody_data output_cody_data.cpp)
target_link_libraries(output_cody_data
  PRIVATE
  Boost::program_options
  LiquidCrystalSystems)

add_executable(compare_rhs_values compare_rhs_values.cpp)
target_link_libraries(compare_rhs_values
  PRIVATE
  Boost::program_options
  LiquidCrystalSystems)

add_executable(plot_lambda_vals plot_lambda_vals.cpp)
target_link_libraries(plot_lambda_vals
  PRIVATE
  Boost::program_options
  HighFive
  LagrangeMultiplier)

# add_executable(read_defect_print_to_grid read_defect_print_to_grid.cpp)
# target_link_libraries(read_defect_print_to_grid
#   deal.II::Interface
#   HighFive
#   Boost::serialization
#   MaierSaupeConstants
#   Postprocessors)

install(TARGETS
  output_cody_data
  compare_rhs_values
  plot_lambda_vals
  # read_defect_print_to_grid
  DESTINATION bin)

install(FILES
  comparing_to_cody_data.py
  plot_data.py
  convert_cody_data_hdf5.py
  find_defect_centers.py
  process_mesh_data.py
  scaling_analysis.py
  weak_scaling_analysis.py
  plot_periodic_singular_potential.py
  DESTINATION bin)
