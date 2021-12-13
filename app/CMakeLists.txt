include(DeclareProgram)

# find dependencies
find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)

add_subdirectory(analysis)
add_subdirectory(examples)
add_subdirectory(simulations)
