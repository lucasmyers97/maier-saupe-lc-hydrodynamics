add_library(LinearInterpolation INTERFACE)

target_include_directories(LinearInterpolation INTERFACE ${SRC_DIR})
target_link_libraries(LinearInterpolation INTERFACE deal.II::Interface)
