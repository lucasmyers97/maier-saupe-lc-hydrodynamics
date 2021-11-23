add_library(BoundaryValues SHARED 
            UniformConfiguration.cpp DefectConfiguration.cpp)
target_include_directories(BoundaryValues PUBLIC ${SRC_DIR})
target_include_directories(BoundaryValues PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(BoundaryValues deal.II::Interface)

install(TARGETS BoundaryValues LIBRARY DESTINATION lib)