add_library(nematic_assembly
  SHARED
  singular_potential_semi_implicit.cpp
  )

target_include_directories(nematic_assembly
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(nematic_assembly
  PUBLIC
  ${SRC_DIR})

target_link_libraries(nematic_assembly
  PUBLIC
  deal.II::Interface
  MaierSaupeConstants
  LagrangeMultiplierAnalytic
  Output)

install(TARGETS nematic_assembly LIBRARY DESTINATION lib)
