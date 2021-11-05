include(DeclareProgram)

# find dependencies
find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(util)
add_subdirectory(examples)
add_subdirectory(simulations)
