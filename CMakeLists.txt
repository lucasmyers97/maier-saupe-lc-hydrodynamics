# Set the minimum version of CMake that's required 
cmake_minimum_required(VERSION 3.12)

# find Deal II, specify with -DDEAL_II_DIR=/path/
find_package(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  REQUIRED)
deal_ii_initialize_cached_variables()
# Deal II has been initialized, so if a target uses it, we should call
#   deal_ii_setup_target(TargetName)

# find Eigen, specify with -DEigen_ROOT=/path/
find_package(Eigen3 REQUIRED)
#   now the target Eigen3::Eigen is defined

# HighFive HDF5 Interface, specify with -DHighFive_ROOT=/path/
find_package(HighFive REQUIRED)
#   now the target HighFive is defined

# CUDA?

project("MSLH")

# library source code
add_subdirectory(src)

# executables
add_subdirectory(app)

# tests
add_subdirectory(test)

