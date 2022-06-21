find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)

add_executable(IsoSteadyStateSim IsoSteadyStateSim.cpp)
target_link_libraries(IsoSteadyStateSim
  PRIVATE
  Boost::program_options
  Boost::serialization
  LiquidCrystalSystems)

add_executable(IsoTimeDependentSim IsoTimeDependentSim.cpp)
target_link_libraries(IsoTimeDependentSim
  PRIVATE
  Boost::program_options
  Boost::serialization
  LiquidCrystalSystems
  SimulationOptions)

add_executable(IsoTimeDependentHydroSim IsoTimeDependentHydroSim.cpp)
target_link_libraries(IsoTimeDependentHydroSim
  PRIVATE
  Boost::program_options
  Boost::serialization
  LiquidCrystalSystems
  SimulationOptions)

add_executable(IsoSteadyStateMPISim IsoSteadyStateMPISim.cpp)
target_link_libraries(IsoSteadyStateMPISim
  PRIVATE
  deal.II::Interface
  Boost::program_options
  Boost::serialization
  LiquidCrystalSystems)

add_executable(IsoTimeDependentMPISim IsoTimeDependentMPISim.cpp)
target_link_libraries(IsoTimeDependentMPISim
  PRIVATE
  Boost::program_options
  Boost::serialization
  LiquidCrystalSystems
  SimulationOptions)

add_executable(bulk_free_energy_calculation bulk_free_energy_calculation.cpp)
target_link_libraries(bulk_free_energy_calculation
  PRIVATE
  Boost::program_options
  HighFive
  deal.II::Interface
  LagrangeMultiplier)

add_executable(HydroFixedConfigurationSim HydroFixedConfigurationSim.cpp)
target_link_libraries(HydroFixedConfigurationSim
  PRIVATE
  deal.II::Interface
  ExampleFunctions
  BasicDrivers)

add_executable(BasicLiquidCrystalSim BasicLiquidCrystalSim.cpp)
target_link_libraries(BasicLiquidCrystalSim
  PRIVATE
  deal.II::Interface
  BasicDrivers)

add_executable(HydroFixedLCSim HydroFixedLCSim.cpp)
target_link_libraries(HydroFixedLCSim
  PRIVATE
  BasicDrivers)

add_executable(NematicSystemMPISim NematicSystemMPISim.cpp)
target_link_libraries(NematicSystemMPISim
  PRIVATE
  deal.II::Interface
  BasicDrivers)

install(TARGETS
  IsoSteadyStateSim
  IsoTimeDependentSim
  IsoTimeDependentHydroSim
  IsoSteadyStateMPISim
  IsoTimeDependentMPISim
  bulk_free_energy_calculation
  HydroFixedConfigurationSim
  BasicLiquidCrystalSim
  HydroFixedLCSim
  NematicSystemMPISim
  DESTINATION bin)
