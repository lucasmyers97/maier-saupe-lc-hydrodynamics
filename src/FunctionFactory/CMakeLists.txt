add_library(FunctionFactory SHARED Function.cpp Factory.cpp)
target_include_directories(FunctionFactory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS FunctionFactory LIBRARY DESTINATION lib)
