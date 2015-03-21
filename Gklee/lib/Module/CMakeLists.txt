file( GLOB SOURCES
      *.cpp
)

add_library( kleeModule STATIC ${SOURCES} )

add_dependencies( kleeModule LLVM )

set_target_properties( kleeModule 
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleeModule" )
