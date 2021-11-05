# shared library for Sphere Lebdev Rule
file(GLOB cpp_src CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[a-zA-Z]*.cpp)
add_library(SphereLebedevRule SHARED ${cpp_src})
target_include_directories(SphereLebedevRule PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS SphereLebedevRule LIBRARY DESTINATION lib)
