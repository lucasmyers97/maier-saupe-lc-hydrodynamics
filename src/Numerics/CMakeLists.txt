find_package(Boost COMPONENTS serialization REQUIRED)

add_library(NumericalTools INTERFACE)
target_include_directories(NumericalTools INTERFACE ${SRC_DIR})

add_library(LagrangeMultiplier SHARED LagrangeMultiplier.cpp)
target_include_directories(LagrangeMultiplier
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LagrangeMultiplier
  PUBLIC ${SRC_DIR})
target_link_libraries(LagrangeMultiplier
  PUBLIC
  deal.II::Interface
  MaierSaupeConstants
  SphereLebedevRule
  Boost::serialization)

add_library(LagrangeMultiplierReduced SHARED LagrangeMultiplierReduced.cpp)
target_include_directories(LagrangeMultiplierReduced
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LagrangeMultiplierReduced
  PUBLIC ${SRC_DIR})

target_link_libraries(LagrangeMultiplierReduced
  PUBLIC
  deal.II::Interface
  MaierSaupeConstants
  SphereLebedevRule
  Boost::serialization)

add_library(LagrangeMultiplierEfficient SHARED LagrangeMultiplierEfficient.cpp)
target_include_directories(LagrangeMultiplierEfficient
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LagrangeMultiplierEfficient
  PUBLIC ${SRC_DIR})

target_link_libraries(LagrangeMultiplierEfficient
  PUBLIC
  deal.II::Interface
  MaierSaupeConstants
  LagrangeMultiplierReduced
  NumericalTools
  Boost::serialization)

add_library(LUMatrix SHARED LUMatrix.cpp)
target_include_directories(LUMatrix
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LUMatrix
  PUBLIC ${SRC_DIR})

install(TARGETS
  LagrangeMultiplier
  LagrangeMultiplierReduced
  LagrangeMultiplierEfficient
  LUMatrix
  LIBRARY DESTINATION lib)
