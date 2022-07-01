add_library(Couplers SHARED
  NematicHydroMPICoupler.cpp)
target_include_directories(Couplers PUBLIC ${SRC_DIR})
target_include_directories(Couplers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Couplers
  PUBLIC
  deal.II::Interface
  LiquidCrystalSystems)

install(TARGETS Couplers LIBRARY DESTINATION lib)
