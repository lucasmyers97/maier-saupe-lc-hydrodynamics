find_package(Boost COMPONENTS program_options serialization headers REQUIRED)

add_library(velocity_field SHARED
    # velocity_field_factory.cpp
    velocity_field.cpp
    uniform_velocity_field.cpp)

target_include_directories(velocity_field PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(velocity_field PUBLIC ${SRC_DIR})

target_link_libraries(velocity_field
    PUBLIC
    deal.II::Interface
    toml
    VectorConversion
    Boost::program_options
    Boost::serialization
    Boost::headers)

install(TARGETS velocity_field LIBRARY DESTINATION lib)
