find_package(Boost COMPONENTS program_options serialization headers REQUIRED)

add_library(BoundaryValues SHARED
    BoundaryValuesFactory.cpp
    BoundaryValues.cpp
    UniformConfiguration.cpp
    PeriodicConfiguration.cpp
    PeriodicSConfiguration.cpp
    DefectConfiguration.cpp
    TwoDefectConfiguration.cpp
    TwistedTwoDefect.cpp
    DzyaloshinskiiFunction.cpp
    MultiDefectConfiguration.cpp
    PerturbativeTwoDefect.cpp
    EscapedRadial.cpp)

target_include_directories(BoundaryValues PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(BoundaryValues PUBLIC ${SRC_DIR})

target_link_libraries(BoundaryValues
    PUBLIC
    deal.II::Interface
    MaierSaupeConstants
    DzyaloshinskiiSystem
    toml
    VectorConversion
    Boost::program_options
    Boost::serialization
    Boost::headers)

install(TARGETS BoundaryValues LIBRARY DESTINATION lib)
