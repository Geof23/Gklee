file( GLOB SOURCES
      *.cpp
)

add_library( kleeSupport STATIC ${SOURCES} )

add_dependencies( kleeSupport LLVM )

set_target_properties( kleeSupport
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleeSupport" )
