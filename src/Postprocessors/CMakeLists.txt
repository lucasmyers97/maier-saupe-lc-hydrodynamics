find_package(HighFive REQUIRED)

add_library(Postprocessors SHARED EvaluateFEObject.cpp)
target_include_directories(Postprocessors PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Postprocessors PUBLIC ${SRC_DIR})

target_link_libraries(Postprocessors
  PRIVATE
  deal.II::Interface
  MaierSaupeConstants
  HighFive
  LagrangeMultiplierAnalytic)

install(TARGETS Postprocessors LIBRARY DESTINATION lib)
