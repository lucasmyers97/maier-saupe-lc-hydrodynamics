declare_program(name IsoSteadyStateSim
                dependencies Boost::serialization
                MaierSaupe
                BoundaryValues
                Postprocessors
                LiquidCrystalSystems)

declare_program(name IsoTimeDependentSim
                dependencies Boost::serialization
                MaierSaupe
                BoundaryValues
                Postprocessors
                LiquidCrystalSystems)

declare_program(name IsoSteadyStateMPISim
                dependencies
                deal.II::Interface
                Boost::serialization
                MaierSaupe
                BoundaryValues
                Postprocessors
                LiquidCrystalSystems)

# not compiling, IDK why
#declare_program(name LdGIsoSteadyState
#                dependencies deal.II::Interface)

declare_program(name bulk_free_energy_calculation
                dependencies Boost::program_options
                             HighFive
                             MaierSaupe)
