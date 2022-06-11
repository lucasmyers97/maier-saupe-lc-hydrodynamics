add_library(ExampleFunctions INTERFACE)
target_include_directories(ExampleFunctions INTERFACE ${SRC_DIR})
target_link_libraries(ExampleFunctions
  INTERFACE
  deal.II::Interface
  MaierSaupeConstants)
