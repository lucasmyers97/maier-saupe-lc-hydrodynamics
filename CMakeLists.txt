# Set the minimum version of CMake that's required 
cmake_minimum_required(VERSION 3.12)

project(maier-saupe-lc-hydrodynamics VERSION 0.1 LANGUAGES C CXX)

# find Deal II
find_package(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  REQUIRED)
# massage Deal II into a more modern CMake Target
if(NOT TARGET deal.II::Interface)
  add_library(deal.II::Interface INTERFACE IMPORTED GLOBAL)
  set_target_properties(deal.II::Interface PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DEAL_II_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${DEAL_II_LIBRARIES_RELEASE}"
    )
endif()

# library source code
add_subdirectory(src)

# executables
add_subdirectory(app)

# tests
add_subdirectory(test)

