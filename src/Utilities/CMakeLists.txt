add_library(LinearInterpolation INTERFACE)
target_include_directories(LinearInterpolation INTERFACE ${SRC_DIR})
target_link_libraries(LinearInterpolation INTERFACE deal.II::Interface)

add_library(Output INTERFACE)
target_include_directories(Output INTERFACE ${SRC_DIR})
target_link_libraries(Output INTERFACE deal.II::Interface)

add_library(MaierSaupeConstants INTERFACE)
target_include_directories(MaierSaupeConstants INTERFACE ${SRC_DIR})

add_library(SimulationOptions SHARED SimulationOptions.cpp)
target_include_directories(SimulationOptions PUBLIC ${SRC_DIR})
target_include_directories(SimulationOptions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(Serialization SHARED Serialization.cpp)
target_include_directories(Serialization PUBLIC ${SRC_DIR})
target_include_directories(Serialization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Serialization
  PUBLIC
  deal.II::Interface)

add_library(DefectGridGenerator INTERFACE)
target_include_directories(DefectGridGenerator INTERFACE ${SRC_DIR})
target_link_libraries(DefectGridGenerator
    INTERFACE
    deal.II::Interface)

add_library(ParameterParser INTERFACE)
target_include_directories(ParameterParser INTERFACE ${SRC_DIR})
target_link_libraries(ParameterParser
    INTERFACE
    deal.II::Interface)

add_library(GridTools GridTools.cpp)
target_include_directories(GridTools 
    PUBLIC 
    ${SRC_DIR})
target_include_directories(GridTools 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(GridTools
    PRIVATE 
    NumericalTools)
target_link_libraries(GridTools
    PUBLIC
    deal.II::Interface)

add_library(VectorConversion
    vector_conversion.cpp
    )

target_include_directories(VectorConversion
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(VectorConversion
    PUBLIC
    deal.II::Interface
    )

install(TARGETS
  SimulationOptions
  Serialization
  GridTools
  VectorConversion
  LIBRARY DESTINATION lib)
