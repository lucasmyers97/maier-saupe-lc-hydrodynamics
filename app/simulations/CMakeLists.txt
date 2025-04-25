find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options headers REQUIRED)

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

# add_executable(BasicLiquidCrystalSim BasicLiquidCrystalSim.cpp)
# target_link_libraries(BasicLiquidCrystalSim
#   PRIVATE
#   deal.II::Interface
#   BasicDrivers)

add_executable(HydroFixedLCSim HydroFixedLCSim.cpp)
target_link_libraries(HydroFixedLCSim
  PRIVATE
  BasicDrivers)

add_executable(NematicSystemMPISim NematicSystemMPISim.cpp)
target_link_libraries(NematicSystemMPISim
  PRIVATE
  deal.II::Interface
  BasicDrivers
  LiquidCrystalSystems
  ParameterParser
  toml
  BoundaryValues
  velocity_field
  Boost::headers)

add_executable(HydroSystemMPISim HydroSystemMPISim.cpp)
target_link_libraries(HydroSystemMPISim
  PRIVATE
  deal.II::Interface
  BasicDrivers)

# add_executable(NematicHydroSystemMPISim NematicHydroSystemMPISim.cpp)
# target_link_libraries(NematicHydroSystemMPISim
#   PRIVATE
#   deal.II::Interface
#   BasicDrivers)

add_executable(periodic_singular_potential periodic_singular_potential.cpp)
target_link_libraries(periodic_singular_potential
  PRIVATE
  HighFive
  deal.II::Interface
  LagrangeMultiplierAnalytic)

add_executable(DzyaloshinskiiSystemSim DzyaloshinskiiSystemSim.cpp)
target_link_libraries(DzyaloshinskiiSystemSim
    PRIVATE
    DzyaloshinskiiSystem)

add_executable(PerturbativeDirectorSystemSim PerturbativeDirectorSystemSim.cpp)
target_link_libraries(PerturbativeDirectorSystemSim
  PRIVATE
  PerturbativeDirectorSystem)

add_executable(ChiralDirectorSystemSim ChiralDirectorSystemSim.cpp)
target_link_libraries(ChiralDirectorSystemSim
  PRIVATE
  toml
  ChiralDirectorSystem)

install(TARGETS
  IsoSteadyStateSim
  IsoTimeDependentSim
  IsoTimeDependentHydroSim
  IsoSteadyStateMPISim
  IsoTimeDependentMPISim
  bulk_free_energy_calculation
  HydroFixedConfigurationSim
  # BasicLiquidCrystalSim
  HydroFixedLCSim
  NematicSystemMPISim
  HydroSystemMPISim
  # NematicHydroSystemMPISim
  periodic_singular_potential
  DzyaloshinskiiSystemSim
  PerturbativeDirectorSystemSim
  ChiralDirectorSystemSim
  DESTINATION bin)
