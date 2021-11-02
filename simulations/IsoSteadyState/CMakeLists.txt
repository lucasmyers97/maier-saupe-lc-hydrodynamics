add_executable(IsoSteadyState IsoSteadyState.cpp)

target_include_directories(IsoSteadyState PRIVATE ${INCLUDE_DIR})
target_link_libraries(IsoSteadyState Boost::serialization maier_saupe_src)
deal_ii_setup_target(IsoSteadyState)

install(TARGETS IsoSteadyState)