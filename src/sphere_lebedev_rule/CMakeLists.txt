# shared library for Sphere Lebdev Rule
add_library(SphereLebedevRule SHARED sphere_lebedev_rule.cpp)
target_include_directories(SphereLebedevRule PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS SphereLebedevRule LIBRARY DESTINATION lib)
