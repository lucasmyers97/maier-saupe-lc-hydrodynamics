find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)

add_library(LiquidCrystalSystems
  SHARED
  IsoTimeDependent.cpp
  IsoTimeDependentHydro.cpp
  IsoSteadyState.cpp
  IsoSteadyStateMPI.cpp
  IsoTimeDependentMPI.cpp)

target_include_directories(LiquidCrystalSystems
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LiquidCrystalSystems
  PUBLIC
  ${SRC_DIR})

target_link_libraries(LiquidCrystalSystems
  PUBLIC
  deal.II::Interface
  HighFive
  MaierSaupeConstants
  LagrangeMultiplier
  LagrangeMultiplierEfficient
  LagrangeMultiplierAnalytic
  BoundaryValues
  Postprocessors
  LinearInterpolation
  Boost::serialization
  Boost::program_options)

add_executable(HydroFixedConfiguration HydroFixedConfiguration.cpp)
target_link_libraries(HydroFixedConfiguration
  PUBLIC
  deal.II::Interface
  MaierSaupeConstants
  SimulationOptions
  Boost::program_options
  LiquidCrystalSystems
  BoundaryValues)

install(TARGETS LiquidCrystalSystems LIBRARY DESTINATION lib)
install(TARGETS HydroFixedConfiguration DESTINATION bin)
