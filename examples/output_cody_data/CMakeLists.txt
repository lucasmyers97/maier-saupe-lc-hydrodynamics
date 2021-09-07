cmake_minimum_required(VERSION 3.21)
project(output_cody_data)
set(CMAKE_CXX_STANDARD 14)

find_package(HighFive REQUIRED)
add_executable(output_cody_data output_cody_data.cpp)
target_link_libraries(output_cody_data HighFive)

add_compile_definitions(DATA_FILE_LOCATION="$ENV{MAIER_SAUPE_DIR}/examples/cody_data/plus-half-defect-cody.hdf5")

find_package(deal.II 9.3.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

deal_ii_initialize_cached_variables()
deal_ii_setup_target(output_cody_data)