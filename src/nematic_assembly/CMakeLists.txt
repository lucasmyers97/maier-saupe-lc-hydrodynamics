add_library(nematic_assembly
  SHARED
  singular_potential_semi_implicit.cpp
  singular_potential_semi_implicit_rotated.cpp
  singular_potential_convex_splitting.cpp
  landau_de_gennes_convex_splitting.cpp
  singular_potential_newtons_method.cpp
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
