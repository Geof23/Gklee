file( GLOB SOURCES
      *.cpp
)

add_library( kleeCore STATIC ${SOURCES} )

add_dependencies( kleeCore LLVM )

set_target_properties( kleeCore 
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleeCore" )
