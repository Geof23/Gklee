include_directories( ./STP/src/STP/include ../llvm/src/LLVM-build/include ../llvm/src/LLVM/include  ${CMAKE_BINARY_DIR}/Gklee/include ./include )

link_directories( ${CMAKE_SOURCE_DIR}/lib )

set( CMAKE_INCLUDE_CURRENT_DIR ON )

add_subdirectory( include/klee/Config )
add_subdirectory( lib )
add_subdirectory( runtime )
add_subdirectory( tools )

