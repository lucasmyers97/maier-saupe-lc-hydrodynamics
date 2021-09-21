cmake_minimum_required(VERSION 3.21)
project(read-defect-to-grid)
set(CMAKE_CXX_STANDARD 14)

include_directories($ENV{MAIER_SAUPE_DIR}/include)

find_package(HighFive REQUIRED)
add_executable(read-defect-print-to-grid read-defect-print-to-grid.cpp)
target_link_libraries(read-defect-print-to-grid HighFive)

find_package(deal.II 9.3.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

deal_ii_initialize_cached_variables()
deal_ii_setup_target(read-defect-print-to-grid)