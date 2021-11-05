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
