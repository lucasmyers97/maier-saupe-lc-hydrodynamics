
# function factory library
# add_subdirectory(Factory)

add_subdirectory(Parameters)

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

# functions for matrix assembly of nematic systems
add_subdirectory(nematic_assembly)

# functions for outputting energy of nematic configuration
add_subdirectory(nematic_energy)

# liquid crystal systems library
add_subdirectory(LiquidCrystalSystems)

# library that couples hydro and nematic evolution
add_subdirectory(Couplers)

# classes which drive the simulations using liquid crystal systems
add_subdirectory(SimulationDrivers)

# GPU accelleration wrapper
#add_subdirectory(cuda)
