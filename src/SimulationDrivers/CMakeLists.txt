add_library(BasicHydroDriver INTERFACE)
target_include_directories(BasicHydroDriver INTERFACE ${SRC_DIR})
target_link_libraries(BasicHydroDriver
  INTERFACE
  deal.II::Interface
  LiquidCrystalSystems)
