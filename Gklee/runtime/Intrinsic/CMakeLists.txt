file ( GLOB SOURCES *.c )

add_library( kleeRuntimeIntrinsic STATIC ${SOURCES} )

add_dependencies ( kleeRuntimeIntrinsic LLVM )

set_target_properties( kleeRuntimeIntrinsic
  PROPERTIES
  OUTPUT_NAME
  kleeRuntimeIntrinsic )

add_custom_command( TARGET kleeRuntimeIntrinsic 
  COMMAND BUILD_BYTECODE_AR( "${CMAKE_C_FLAGS} -I. -I../../include ${LLVM_SOURCE_DIR}/include" ${SOURCES} ${GKLEE_LIB}/kleeRuntimeIntrinsic.bca . )
)

