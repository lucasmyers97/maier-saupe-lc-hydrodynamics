cmake_minimum_required(VERSION 3.20)

# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-12")
# set(CMAKE_CUDA_COMPILER "/usr/bin/clang++-12")
# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75")
# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75 -x cuda -lcudart_static -ldl -lrt -pthread")
project(main_test VERSION 1.0.0
        LANGUAGES CXX CUDA)

set(home_dir "$ENV{MAIER_SAUPE_DIR}")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(main_test main_test.cpp)
set_target_properties(main_test PROPERTIES CUDA_ARCHITECTURES "75")
set_target_properties(main_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main_test ${Boost_LIBRARIES})
include_directories("${home_dir}/include")

add_subdirectory(lagrange_gpu_test)