add_library(LinearInterpolation INTERFACE)
target_include_directories(LinearInterpolation INTERFACE ${SRC_DIR})
target_link_libraries(LinearInterpolation INTERFACE deal.II::Interface)

add_library(MaierSaupeConstants INTERFACE)
target_include_directories(MaierSaupeConstants INTERFACE ${SRC_DIR})

add_library(SimulationOptions SimulationOptions.cpp)
target_include_directories(SimulationOptions PUBLIC ${SRC_DIR})
target_include_directories(SimulationOptions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS SimulationOptions LIBRARY DESTINATION lib)
