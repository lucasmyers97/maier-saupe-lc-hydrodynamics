# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75 -Wno-unknown-cuda-version")
# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75 -lcudart_static -ldl -lrt -pthread")
#set_target_properties(main_test PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(main_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set the minimum version of CMake that's required
cmake_minimum_required(VERSION 3.12)

project(maier-saupe-lc-hydrodynamics VERSION 0.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_COMPILER "clang++-13")
set(CMAKE_CXX_COMPILER "g++" CACHE STRING "Compiler")
add_compile_options(-Wall -Wextra -pedantic)


if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Need to set -DCMAKE_BUILD_TYPE")
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Load additional macros used by this project
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

# find_package(Trilinos
#   HINTS ${TRILINOS_DIR}
#   REQUIRED)

# find Deal II
find_package(deal.II 9.4.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  REQUIRED)
# massage Deal II into a more modern CMake Target
if(NOT TARGET deal.II::Interface)
    # if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    #   add_library(deal.II::Interface ALIAS dealii::dealii_debug)
    # else()
    #   add_library(deal.II::Interface ALIAS dealii::dealii_release)
    # endif()
  add_library(deal.II::Interface INTERFACE IMPORTED GLOBAL)
  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set_target_properties(deal.II::Interface PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${DEAL_II_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${DEAL_II_LIBRARIES_DEBUG}"
      )
  else()
    set_target_properties(deal.II::Interface PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${DEAL_II_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${DEAL_II_LIBRARIES_RELEASE}"
      )
  endif()
  # target_link_libraries(deal.II::Interface INTERFACE ${Trilinos_LIBRARIES})
endif()

# -----------------------------------------------------------------------------
# Get all propreties that cmake supports
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)
endif()
    
function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()
    
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()
# -----------------------------------------------------------------------------
# print_target_properties(dealii::interface_trilinos)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# include(cmake/CheckGit.cmake)
# CheckGitSetup()

# library source code
add_subdirectory(src)

# executables
add_subdirectory(app)

# tests
add_subdirectory(test)
