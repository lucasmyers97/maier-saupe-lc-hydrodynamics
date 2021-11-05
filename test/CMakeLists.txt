
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++-12")
# set(CMAKE_CUDA_COMPILER "/usr/bin/clang++-12")
# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75 -Wno-unknown-cuda-version")
# set(CMAKE_CUDA_FLAGS "--cuda-path=/usr/local/cuda-11/lib64 --cuda-gpu-arch=sm_75 -lcudart_static -ldl -lrt -pthread")
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_executable(main_test 
  main_test.cpp
  lagrange_print_lambda.cpp
  )
#set_target_properties(main_test PROPERTIES CUDA_ARCHITECTURES "75")
#set_target_properties(main_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main_test PRIVATE Boost::unit_test_framework MaierSaupe)
