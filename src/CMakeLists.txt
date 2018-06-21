# generate templated sources with m4
add_custom_command(OUTPUT replicator.m4
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/replicator.m4 ./
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/replicator.m4
)

set(M4_SRC
  fcudaMemcpy_function.F90
  fcudaMemcpyAsync_function.F90
  fcudaHostRegister_function.F90
)
foreach(m4src ${M4_SRC})
  add_custom_command(OUTPUT ${m4src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${m4src}.m4 ./
    COMMAND m4 ${m4src}.m4 > ${m4src}
    MAIN_DEPENDENCY ${m4src}.m4
    DEPENDS replicator.m4
  )
endforeach(m4src)

# build library
set(SRC
  f90_assert.F90
  cuda_c_binding.F90
  ${M4_SRC}
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
