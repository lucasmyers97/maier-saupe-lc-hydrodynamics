find_package(Boost COMPONENTS serialization REQUIRED)

add_library(BasicDrivers SHARED BasicLiquidCrystalDriver.cpp)
target_include_directories(BasicDrivers PUBLIC ${SRC_DIR})
target_link_libraries(BasicDrivers
  PUBLIC
  deal.II::Interface
  LiquidCrystalSystems
  Boost::serialization)
