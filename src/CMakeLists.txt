set(SRC
  f90_assert.F90
  cuda_c_binding.F90
  fcuda.F90
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
