find_package(Boost COMPONENTS serialization program_options REQUIRED)

# helper library for Sphere Lebedev Rules
add_subdirectory(sphere_lebedev_rule)

# function factory library
add_subdirectory(Factory)

# boundary values library
add_subdirectory(BoundaryValues)

# postprocessors library
add_subdirectory(Postprocessors)

# liquid crystal systems library
add_subdirectory(LiquidCrystalSystems)

# TODO: break up maier-saupe library into other, smaller libraries
# shared library for Maier Saupe Liquid Crystal simulations
file(GLOB cpp_src CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[a-zA-Z]*.cpp)

add_library(MaierSaupe SHARED ${cpp_src})
target_include_directories(MaierSaupe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MaierSaupe
  SphereLebedevRule deal.II::Interface Boost::serialization)

install(TARGETS MaierSaupe LIBRARY DESTINATION lib)

# GPU accelleration wrapper
#add_subdirectory(cuda)
