file( GLOB SOURCES
      *.cpp
)

add_library( kleeBasic STATIC ${SOURCES} )

add_dependencies( kleeBasic LLVM )

set_target_properties( kleeBasic 
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleeBasic" )

