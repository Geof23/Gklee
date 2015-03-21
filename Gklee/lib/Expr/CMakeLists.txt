file( GLOB SOURCES
      *.cpp
)

add_library( kleaverExpr STATIC ${SOURCES} )

add_dependencies( kleaverExpr LLVM )

message( "CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS} in ${CMAKE_CURRENT_SOURCE_DIR}" )

set_target_properties( kleaverExpr 
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleaverExpr" )

