find_package(Boost COMPONENTS serialization REQUIRED)

add_library(BasicDrivers SHARED
  BasicLiquidCrystalDriver.cpp
  NematicSystemMPIDriver.cpp
  NematicHydroSystemMPIDriver.cpp)
target_include_directories(BasicDrivers PUBLIC ${SRC_DIR})
target_link_libraries(BasicDrivers
  PUBLIC
  deal.II::Interface
  LiquidCrystalSystems
  velocity_field
  Couplers
  SetDefectBoundaryConstraints
  Boost::serialization
  Serialization)

install(TARGETS BasicDrivers LIBRARY DESTINATION lib)
