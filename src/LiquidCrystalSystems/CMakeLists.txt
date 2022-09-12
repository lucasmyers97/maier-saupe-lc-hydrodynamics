find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)

add_library(LiquidCrystalSystems
  SHARED
  HydroFixedConfiguration.cpp
  LiquidCrystalSystem.cpp
  NematicSystemMPI.cpp
  HydroSystemMPI.cpp
  IsoTimeDependent.cpp
  IsoTimeDependentHydro.cpp
  IsoSteadyState.cpp
  IsoSteadyStateMPI.cpp
  IsoTimeDependentMPI.cpp
  )

target_include_directories(LiquidCrystalSystems
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LiquidCrystalSystems
  PUBLIC
  ${SRC_DIR})

target_link_libraries(LiquidCrystalSystems
  PUBLIC
  deal.II::Interface
  MaierSaupeConstants
  LagrangeMultiplier
  LagrangeMultiplierAnalytic
  BoundaryValues
  Postprocessors
  LinearInterpolation
  Boost::serialization
  Boost::program_options
  Output)

target_link_libraries(LiquidCrystalSystems
  PRIVATE
  HighFive)

add_library(DzyaloshinskiiSystem
    DzyaloshinskiiSystem.cpp)

target_include_directories(DzyaloshinskiiSystem
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(DzyaloshinskiiSystem
    PUBLIC
    ${SRC_DIR})

target_link_libraries(DzyaloshinskiiSystem
    PUBLIC
    deal.II::Interface)

install(TARGETS LiquidCrystalSystems LIBRARY DESTINATION lib)
install(TARGETS DzyaloshinskiiSystem LIBRARY DESTINATION lib)
