set( SOURCES intrinsics.c )

add_library( kleeRuntest_d SHARED ${SOURCES} )
add_library( kleeRuntest_s STATIC ${SOURCES} )

add_dependencies( kleeRuntest_d LLVM )
add_dependencies( kleeRuntest_s LLVM )

set_target_properties( kleeRuntest_s
		       PROPERTIES
		       OUTPUT_NAME
		       "kleeRuntest" )

set_target_properties( kleeRuntest_d
		       PROPERTIES
		       OUTPUT_NAME
		       "kleeRuntest" )
