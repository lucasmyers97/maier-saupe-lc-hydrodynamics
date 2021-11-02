# helpful cmake package for custom macros
include(CMakeParseArguments)

#################################################################################################
# CMake macro 'declare_program'
# Declare a new program
# Arguments:
#   name - name of program, same as name of source file but without .cpp extension
#   dependencies - CMake target dependencies
#################################################################################################
macro(declare_program)

  #set(options ) # options are parameters with no values
  set(oneValueArgs name) #single value parameters
  set(multiValueArgs dependencies) #multiple value parameters
  cmake_parse_arguments(declare_program "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  add_executable(${declare_program_name} ${declare_program_name}.cpp)
  #deal_ii_setup_target(${declare_program_name})
  target_link_libraries(${declare_program_name} PRIVATE ${declare_program_dependencies})
  install(TARGETS ${declare_program_name} DESTINATION bin)
endmacro()

# find dependencies
find_package(HighFive REQUIRED)
find_package(Boost COMPONENTS serialization program_options REQUIRED)
find_package(Eigen3)

if(Eigen3_FOUND)
  declare_program(name eigen_test
         dependencies Eigen3::Eigen)
endif()

declare_program(name IsoSteadyState
                dependencies Boost::serialization
                             MaierSaupe)

# not compiling, IDK why
#declare_program(name LdGIsoSteadyState
#                dependencies deal.II::Interface)

declare_program(name bulk_free_energy_calculation
                dependencies HighFive
                             MaierSaupe)
