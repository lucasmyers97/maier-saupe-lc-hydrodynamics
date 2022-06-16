add_library(BasicDrivers INTERFACE)
target_include_directories(BasicDrivers INTERFACE ${SRC_DIR})
target_link_libraries(BasicDrivers
  INTERFACE
  deal.II::Interface
  LiquidCrystalSystems)
