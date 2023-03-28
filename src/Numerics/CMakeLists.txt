find_package(Boost COMPONENTS serialization REQUIRED)

add_library(NumericalTools INTERFACE)
target_include_directories(NumericalTools INTERFACE ${SRC_DIR})
target_link_libraries(NumericalTools
  INTERFACE
  deal.II::Interface)

add_library(MatrixTools INTERFACE)
target_include_directories(MatrixTools INTERFACE ${SRC_DIR})
target_link_libraries(MatrixTools
  INTERFACE
  deal.II::Interface)

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

add_library(SetDefectBoundaryConstraints INTERFACE)
target_include_directories(SetDefectBoundaryConstraints
    INTERFACE
    ${SRC_DIR})
target_link_libraries(SetDefectBoundaryConstraints
    INTERFACE
    deal.II::Interface
    MaierSaupeConstants)

# add_library(DzyaloshinskiiFunction INTERFACE)
# target_include_directories(DzyaloshinskiiFunction
#     INTERFACE 
#     ${SRC_DIR})
# target_link_libraries(DzyaloshinskiiFunction
#     INTERFACE
#     DzyaloshinskiiSystem)

# add_library(LagrangeMultiplierEfficient SHARED LagrangeMultiplierEfficient.cpp)
# target_include_directories(LagrangeMultiplierEfficient
#   PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# target_include_directories(LagrangeMultiplierEfficient
#   PUBLIC ${SRC_DIR})
# target_link_libraries(LagrangeMultiplierEfficient
#   PUBLIC
#   deal.II::Interface
#   MaierSaupeConstants
#   LagrangeMultiplierReduced
#   NumericalTools
#   Boost::serialization)

add_library(LagrangeMultiplierAnalytic SHARED LagrangeMultiplierAnalytic.cpp)
target_include_directories(LagrangeMultiplierAnalytic
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(LagrangeMultiplierAnalytic
  PUBLIC ${SRC_DIR})
target_link_libraries(LagrangeMultiplierAnalytic
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
  #  LagrangeMultiplierEfficient
  LagrangeMultiplierAnalytic
  LUMatrix
  LIBRARY DESTINATION lib)
