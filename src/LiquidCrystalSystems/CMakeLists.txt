find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)

add_library(LiquidCrystalSystems SHARED IsoTimeDependent.cpp IsoSteadyState.cpp IsoSteadyStateMPI.cpp)

target_include_directories(LiquidCrystalSystems
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LiquidCrystalSystems PUBLIC ${SRC_DIR})

target_link_libraries(LiquidCrystalSystems
  deal.II::Interface
  HighFive
  MaierSaupe
  Postprocessors
  BoundaryValues
  LinearInterpolation
  Boost::serialization
  Boost::program_options)

install(TARGETS LiquidCrystalSystems LIBRARY DESTINATION lib)
