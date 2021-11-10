

# declare_program(name IsoSteadyState
#                 dependencies Boost::serialization
#                              MaierSaupe)

# not compiling, IDK why
#declare_program(name LdGIsoSteadyState
#                dependencies deal.II::Interface)

declare_program(name bulk_free_energy_calculation
                dependencies Boost::program_options
                             HighFive
                             MaierSaupe)
