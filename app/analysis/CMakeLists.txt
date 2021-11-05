include(DeclareProgram)

declare_program(name read-defect-print-to-grid
                dependencies HighFive
                             MaierSaupe)

install(FILES comparing-to-cody-data.py plot_data.py DESTINATION bin)

# not compiling because "DATA_FILE_LOCATION" is not defined
#declare_program(name output-cody-data
#                dependencies HighFive
#                             MaierSaupe)
