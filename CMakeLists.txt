# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75 -Wno-unknown-cuda-version")
# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75 -lcudart_static -ldl -lrt -pthread")
#set_target_properties(main_test PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(main_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set the minimum version of CMake that's required
cmake_minimum_required(VERSION 3.12)

project(maier-saupe-lc-hydrodynamics VERSION 0.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-13")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Load additional macros used by this project
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

# find_package(Trilinos
#   HINTS ${TRILINOS_DIR}
#   REQUIRED)

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
  # target_link_libraries(deal.II::Interface INTERFACE ${Trilinos_LIBRARIES})
endif()


set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# library source code
add_subdirectory(src)

# executables
add_subdirectory(app)

# tests
add_subdirectory(test)
