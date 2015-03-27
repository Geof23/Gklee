file( GLOB SOURCES
      *.c
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library( kleeRuntimePOSIX STATIC ${SOURCES} )

add_dependencies ( kleeRuntimePOSIX LLVM )

set( INTRINSIC_BCA libkleeRuntimePOSIX.bca )

add_custom_command( TARGET kleeRuntimePOSIX 
  PRE_LINK
  COMMAND ${CMAKE_COMMAND} -DBINARIES=${CMAKE_BINARY_DIR} -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR} -DDEST=${CMAKE_SOURCE_DIR}/lib/${INTRINSIC_BCA} -DBBBIN=${CMAKE_SOURCE_DIR}/llvm/src/LLVM-build/bin -DHOME=${CMAKE_SOURCE_DIR} -P ${CMAKE_MODULE_PATH}/BuildBCArch.cmake
  COMMENT "launching BuildBCArch.cmake"
)

set_target_properties( kleeRuntimePOSIX 
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleeRuntimePOSIX" )

