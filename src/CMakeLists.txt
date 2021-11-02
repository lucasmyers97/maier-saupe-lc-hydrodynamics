# helper library for Sphere Lebdev Rules
add_subdirectory(sphere_lebdev_rule)

# shared library for Maier Saupe Liquid Crystal simulations
file(GLOB cpp_src CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[a-zA-Z]*.cpp)

add_library(MaierSaupe SHARED ${cpp_src})
target_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MairSaupe SphereLebdevRule)

set_target_properties(MaierSaupe
  PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS MaierSaupe LIBRARY DESTINATION lib)

# GPU accelleration wrapper
add_subdirectory(cuda)
