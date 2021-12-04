find_package(Boost COMPONENTS program_options serialization REQUIRED)

add_library(BoundaryValues SHARED
            UniformConfiguration.cpp DefectConfiguration.cpp)
target_include_directories(BoundaryValues PUBLIC ${SRC_DIR})
target_include_directories(BoundaryValues PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(BoundaryValues deal.II::Interface
                      Boost::program_options Boost::serialization)

install(TARGETS BoundaryValues LIBRARY DESTINATION lib)
