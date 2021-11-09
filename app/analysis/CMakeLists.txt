include(DeclareProgram)

declare_program(name read-defect-print-to-grid
                dependencies HighFive
                             MaierSaupe)

install(FILES comparing-to-cody-data.py plot_data.py convert_cody_data_hdf5.py
        DESTINATION bin)

# declare_program(name output-cody-data
#                dependencies HighFive MaierSaupe)
