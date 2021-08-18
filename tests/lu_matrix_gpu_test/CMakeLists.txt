file(GLOB lu_inversion_gpu_sources "*.cu")
target_sources(main_test PRIVATE "${lu_inversion_gpu_sources}" 
               "${home_dir}/src/LUMatrixGPU.cu" 
               "${home_dir}/extern-src/sphere_lebedev_rule.cpp")