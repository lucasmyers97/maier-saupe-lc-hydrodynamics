cmake_minimum_required(VERSION 3.12)

project(maier-saupe-lc-hydrodynamics VERSION 0.1 LANGUAGES C CXX CUDA)

# Read in external packages
find_package(Boost 1.76 COMPONENTS program_options serialization REQUIRED)
find_package(HighFive REQUIRED)
find_package(deal.II 9.3.0 REQUIRED)
deal_ii_initialize_cached_variables()

# Set directory variables
set(LOCAL_INCLUDE_DIR /usr/local/include)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(EXTERN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern-src)

# Get files from source directories for project-wide library
set(SOURCE_FILES 
    ${SOURCE_DIR}/LagrangeMultiplier.cpp
    ${SOURCE_DIR}/LUMatrix.cpp)
set(EXTERN_SOURCE_FILES 
    ${EXTERN_SOURCE_DIR}/sphere_lebedev_rule.cpp)

# Create project library and appropriately link external libraries
add_library(maier_saupe_src ${SOURCE_FILES} ${EXTERN_SOURCE_FILES})
target_include_directories(maier_saupe_src PRIVATE ${INCLUDE_DIR})
deal_ii_setup_target(maier_saupe_src)

# Add macro for saving data files in an appropriate way
macro(set_data_path)
    set(data_path ${CMAKE_SOURCE_DIR}/data)
    file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    string(TIMESTAMP date_time "%Y-%m-%d_%H-%M-%S")
    set(cur_data_path ${data_path}/${rel_path}/${date_time})
#    if (NOT IS_DIRECTORY ${cur_data_path})
#        file(MAKE_DIRECTORY ${cur_data_path})
#    endif()
endmacro()

install(TARGETS maier_saupe_src)

add_subdirectory(scripts)
add_subdirectory(simulations)