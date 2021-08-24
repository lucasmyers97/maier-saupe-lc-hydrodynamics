cmake_minimum_required(VERSION 3.21)
project(read_cody_data)
set(CMAKE_CXX_STANDARD 11)

find_package(HighFive REQUIRED)
add_executable(read_cody_data read_cody_data.cpp)
target_link_libraries(read_cody_data HighFive)

add_compile_definitions(DATA_FILE_LOCATION="$ENV{MAIER_SAUPE_DIR}/examples/cody_data/plus-half-defect-cody.hdf5")