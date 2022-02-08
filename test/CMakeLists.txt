find_package(HighFive REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework serialization)

add_executable(main_test
  main_test.cpp
  lagrange_multiplier_test.cpp
  # factory_test.cpp
  boundary_values_test.cpp
  iso_steady_state_test.cpp
  )

target_link_libraries(main_test
  PRIVATE
  Boost::unit_test_framework
  Boost::serialization
  LagrangeMultiplier
  SphereLebedevRule
  MaierSaupeConstants
  # Factory
  BoundaryValues
  LiquidCrystalSystems)
install(TARGETS main_test DESTINATION bin/tests)
