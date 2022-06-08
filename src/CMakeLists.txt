
# function factory library
# add_subdirectory(Factory)

# helper library for lebedev integration
add_subdirectory(sphere_lebedev_rule)

# utilities just helps with various operations
add_subdirectory(Utilities)

# numerical support (e.g. LagrangeMultipliers)
add_subdirectory(Numerics)

# random functions that might be used in a simulation
# (e.g. forcing function from +1/2 active defect)
add_subdirectory(ExampleFunctions)

# boundary values library
add_subdirectory(BoundaryValues)

# postprocessors library
add_subdirectory(Postprocessors)

# liquid crystal systems library
add_subdirectory(LiquidCrystalSystems)

# GPU accelleration wrapper
#add_subdirectory(cuda)
