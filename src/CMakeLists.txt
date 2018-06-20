set(SRC
  f90_assert.F90
  cuda_c_binding.F90
  fcudaMemcpy_function.F90
  fcuda_register_functions.F90
  fcuda.F90
)

# fcudaMemcpy_function.F90 is generated via m4
add_custom_command(OUTPUT fcudaMemcpy_function.F90
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/fcudaMemcpy_function.F90.m4 ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND m4 fcudaMemcpy_function.F90.m4 > fcudaMemcpy_function.F90
  MAIN_DEPENDENCY fcudaMemcpy_function.F90.m4
)

add_library(fcuda ${SRC})

set(LIB_MOD_DIR ${CMAKE_CURRENT_BINARY_DIR}/mod_files/)
set_target_properties(fcuda PROPERTIES Fortran_MODULE_DIRECTORY ${LIB_MOD_DIR})

target_link_libraries(fcuda "${CUDA_LIBRARIES}")

target_include_directories(fcuda PUBLIC
    $<BUILD_INTERFACE:${LIB_MOD_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS fcuda
        EXPORT  fcuda
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
install(DIRECTORY ${LIB_MOD_DIR} DESTINATION include)

if(ENABLE_TESTS)
  add_subdirectory(test)
endif()
