find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_executable(main_test 
  main_test.cpp
  lagrange_print_lambda.cpp
  )
  
target_link_libraries(main_test PRIVATE Boost::unit_test_framework MaierSaupe)
