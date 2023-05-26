add_library(nematic_energy
  SHARED
  singular_potential_energy.cpp
  )

target_include_directories(nematic_energy
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(nematic_energy
  PUBLIC
  ${SRC_DIR})

target_link_libraries(nematic_energy
  PUBLIC
  deal.II::Interface
  LagrangeMultiplierAnalytic)

install(TARGETS nematic_energy LIBRARY DESTINATION lib)
