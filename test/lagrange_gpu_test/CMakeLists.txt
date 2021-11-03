# file(GLOB lagrange_gpu_sources "*.cu")
list(APPEND lagrange_gpu_sources "${CMAKE_CURRENT_SOURCE_DIR}/calc_exp_lambda_test.cu")
list(APPEND lagrange_gpu_sources "${CMAKE_CURRENT_SOURCE_DIR}/calc_lambda_test.cu")
list(APPEND lagrange_gpu_sources "${CMAKE_CURRENT_SOURCE_DIR}/calc_res_jac_test.cu")
list(APPEND lagrange_gpu_sources "${CMAKE_CURRENT_SOURCE_DIR}/initialize_inversion_test.cu")
list(APPEND lagrange_gpu_sources "${CMAKE_CURRENT_SOURCE_DIR}/lagrange_integral_test.cu")
list(APPEND lagrange_gpu_sources "${CMAKE_CURRENT_SOURCE_DIR}/set_lebedev_test.cu")
target_sources(main_test PRIVATE "${lagrange_gpu_sources}" 
               "${home_dir}/extern-src/sphere_lebedev_rule.cpp")
