file( GLOB SOURCES
      *.c
)

add_library( cudaRuntimeIntrinsic STATIC ${SOURCES} )

add_dependencies( cudaRuntimeIntrinsic LLVM )

set_target_properties( cudaRuntimeIntrinsic 
		       PROPERTIES 
		       OUTPUT_NAME
		       "cudaRuntimeIntrinsic" )

# get_property( our_includes DIRECTORY . PROPERTY INCLUDE_DIRECTORIES )
# set( INCS "" )
# foreach( DIR ${our_includes} )
#   set( INCS "${INCS} -I${DIR} " )
# endforeach( DIR )

# set( FLAGS ${CMAKE_CXX_FLAGS}${INCS} )

# message( "in ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt:")
# message( "FLAGS is ${FLAGS} " )
# message( "SOURCES is ${SOURCES}" )

# add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cudaRuntimeIntrinsic.bca
#   DEPENDS CLANG LLVM
#   COMMAND BUILD_BYTECODE_AR( ${FLAGS} ${SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/cudaRuntimeIntrinsic.bca ${CMAKE_CURRENT_SOURCE_DIR} )
# )

