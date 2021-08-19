file(GLOB lagrange_gpu_sources "*.cu")
target_sources(main_test PRIVATE "${lagrange_gpu_sources}" 
               "${home_dir}/extern-src/sphere_lebedev_rule.cpp")
