# shared library for Sphere Lebdev Rule
add_library(SphereLebedevRule SHARED sphere_lebedev_rule.cpp)
target_include_directories(SphereLebedevRule PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(SphereLebedevRule PUBLIC ${SRC_DIR})
install(TARGETS SphereLebedevRule LIBRARY DESTINATION lib)
