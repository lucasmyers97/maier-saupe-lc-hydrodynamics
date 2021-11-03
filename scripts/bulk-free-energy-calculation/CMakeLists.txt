add_executable(bulk_free_energy_calculation bulk_free_energy_calculation.cpp)

target_include_directories(bulk_free_energy_calculation PRIVATE ${INCLUDE_DIR})
target_link_libraries(bulk_free_energy_calculation 
                      HighFive Boost::program_options maier_saupe_src)
deal_ii_setup_target(bulk_free_energy_calculation)

install(TARGETS bulk_free_energy_calculation)