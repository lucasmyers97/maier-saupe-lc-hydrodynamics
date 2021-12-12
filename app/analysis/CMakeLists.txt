include(DeclareProgram)

# declare_program(name read-defect-print-to-grid
#                 dependencies HighFive
#                 MaierSaupe
#                 Boost::serialization
#                 LiquidCrystalSystems)

add_executable(output-cody-data output-cody-data.cpp)
target_link_libraries(output-cody-data
  LinearInterpolation
  Postprocessors
  HighFive
  deal.II::Interface
  Boost::program_options)

install(TARGETS output-cody-data DESTINATION bin)

install(FILES comparing-to-cody-data.py plot_data.py convert_cody_data_hdf5.py
        DESTINATION bin)

# declare_program(name output-cody-data
#                dependencies HighFive MaierSaupe)
