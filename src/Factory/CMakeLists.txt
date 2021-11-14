add_library(Factory SHARED Function.cpp)
target_include_directories(Factory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS Factory LIBRARY DESTINATION lib)
