file( GLOB SOURCES
      *.cpp
)

add_library( kleaverSolver STATIC ${SOURCES} )

add_dependencies( kleaverSolver LLVM STP )

set_target_properties( kleaverSolver 
		       PROPERTIES 
		       OUTPUT_NAME
		       "kleaverSolver" )
