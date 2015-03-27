file( GLOB SOURCES
      *.c
)

add_library( cudaRuntimeIntrinsic STATIC ${SOURCES} )

add_dependencies( cudaRuntimeIntrinsic LLVM )

set( INTRINSIC_BCA libcudaRuntimeIntrinsic.bca )

add_custom_command( TARGET cudaRuntimeIntrinsic 
  PRE_LINK
  COMMAND ${CMAKE_COMMAND} -DBINARIES=${CMAKE_BINARY_DIR} -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR} -DDEST=${CMAKE_SOURCE_DIR}/lib/${INTRINSIC_BCA} -DBBBIN=${CMAKE_SOURCE_DIR}/llvm/src/LLVM-build/bin -DHOME=${CMAKE_SOURCE_DIR} -P ${CMAKE_MODULE_PATH}/BuildBCArch.cmake
  COMMENT "launching BuildBCArch.cmake"
)

set_target_properties( cudaRuntimeIntrinsic 
		       PROPERTIES 
		       OUTPUT_NAME
		       cudaRuntimeIntrinsic )

