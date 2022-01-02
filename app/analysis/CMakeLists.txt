include(DeclareProgram)

# declare_program(name read-defect-print-to-grid
#                 dependencies HighFive
#                 MaierSaupe
#                 Boost::serialization
#                 LiquidCrystalSystems)

add_executable(output-cody-data output-cody-data.cpp)
target_link_libraries(output-cody-data
  Boost::program_options
  LiquidCrystalSystems)

add_executable(compare_rhs_values compare_rhs_values.cpp)
target_link_libraries(compare_rhs_values
  Boost::program_options
  LiquidCrystalSystems)

add_executable(plot_lambda_vals plot_lambda_vals.cpp)
target_link_libraries(plot_lambda_vals
  Boost::program_options
  deal.II::Interface
  HighFive
  MaierSaupe)

install(TARGETS output-cody-data compare_rhs_values plot_lambda_vals DESTINATION bin)

install(FILES comparing-to-cody-data.py plot_data.py convert_cody_data_hdf5.py
  find_defect_centers.py process_mesh_data.py
  DESTINATION bin)

# declare_program(name output-cody-data
#                dependencies HighFive MaierSaupe)
