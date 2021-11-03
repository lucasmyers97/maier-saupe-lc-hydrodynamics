add_executable(read-defect-print-to-grid read-defect-print-to-grid.cpp)

target_include_directories(read-defect-print-to-grid PRIVATE ${INCLUDE_DIR})
target_link_libraries(read-defect-print-to-grid HighFive maier_saupe_src)
deal_ii_setup_target(read-defect-print-to-grid)

install(TARGETS read-defect-print-to-grid)