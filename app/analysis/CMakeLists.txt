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

add_executable(calc_lambda calc_lambda.cpp)
target_link_libraries(calc_lambda
    PRIVATE
    HighFive
    LagrangeMultiplierAnalytic
    deal.II::Interface)

add_executable(get_points_around_defects get_points_around_defects.cpp)
target_link_libraries(get_points_around_defects
    PRIVATE
    Boost::program_options
    deal.II::Interface
    BasicDrivers)

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
  calc_lambda
  get_points_around_defects
  # read_defect_print_to_grid
  DESTINATION bin)
