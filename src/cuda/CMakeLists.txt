# shared library for Maier Saupe Liquid Crystal simulations
file(GLOB cuda_src CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[a-zA-Z]*.cu)

add_library(MaierSaupeGPU SHARED ${cuda_src})
target_include_directories(MaierSaupeGPU PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MaierSaupeGPU MaierSaupe)
install(TARGETS MaierSaupeGPU LIBRARY DESTINATION lib)
