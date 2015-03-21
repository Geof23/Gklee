file( GLOB SOURCES
      *.c
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library( kleeRuntimePOSIX STATIC ${SOURCES} )

add_dependencies ( kleeRuntimePOSIX LLVM )

set_target_properties( kleeRuntimePOSIX 
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleeRuntimePOSIX" )

# get_property( our_includes DIRECTORY . PROPERTY INCLUDE_DIRECTORIES )
# set( INCS "" )
# foreach( DIR ${our_includes} )
#   set( INCS "${INCS} -I${DIR} " )
# endforeach( DIR )

# set( FLAGS ${CMAKE_CXX_FLAGS}${INCS} )

# message( "in ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt:")
# message( "FLAGS is ${FLAGS} " )
# message( "SOURCES is ${SOURCES}" )

# add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kleeRuntimePOSIX.bca
#   DEPENDS CLANG LLVM
#   COMMAND BUILD_BYTECODE_AR( ${FLAGS} ${SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/kleeRuntimePOSIX.bca ${CMAKE_CURRENT_SOURCE_DIR} )
# )
