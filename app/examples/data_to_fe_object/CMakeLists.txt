cmake_minimum_required(VERSION 3.21)
project(data_to_fe_object)
set(CMAKE_CXX_STANDARD 14)

include_directories($ENV{MAIER_SAUPE_DIR}/include)

find_package(HighFive REQUIRED)
add_executable(data_to_fe_object data_to_fe_object.cpp)
target_link_libraries(data_to_fe_object HighFive)

add_compile_definitions(DATA_FILE_LOCATION="$ENV{MAIER_SAUPE_DIR}/examples/cody_data/plus-half-defect-cody.hdf5")

find_package(deal.II 9.3.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

deal_ii_initialize_cached_variables()
deal_ii_setup_target(data_to_fe_object)