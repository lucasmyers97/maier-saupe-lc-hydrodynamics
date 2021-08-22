file(GLOB lagrange_wrapper_sources "*.cu")
target_sources(main_test PRIVATE "${lagrange_wrapper_sources}" 
               "${home_dir}/src/LagrangeGPUWrapper.cu"
               "${home_dir}/extern-src/sphere_lebedev_rule.cpp")