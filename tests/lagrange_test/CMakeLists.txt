file(GLOB lagrange_sources "*.cpp")
target_sources(main_test PRIVATE "${lagrange_sources}" 
               "${home_dir}/src/LagrangeMultiplier.cpp"
               "${home_dir}/extern-src/sphere_lebedev_rule.cpp")

find_package(deal.II 9.0.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
deal_ii_initialize_cached_variables()
deal_ii_setup_target(main_test)