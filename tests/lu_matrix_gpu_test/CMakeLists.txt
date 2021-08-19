file(GLOB lu_inversion_gpu_sources "*.cu")
target_sources(main_test PRIVATE "${lu_inversion_gpu_sources}")