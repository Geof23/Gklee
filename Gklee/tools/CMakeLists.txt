include_directories( ../stp/src/STP/include ${CMAKE_CURRENT_SOURCE_DIR}/../../llvm/src/LLVM-build/include ${CMAKE_CURRENT_SOURCE_DIR}/../../llvm/src/LLVM/include  . ../include )

add_definitions( "-DKLEE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/..\"" )

link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../../llvm/src/LLVM-build/lib 
  ${CMAKE_CURRENT_SOURCE_DIR}/../stp/lib)

add_subdirectory( kleaver )
add_subdirectory( klee )
add_subdirectory( klee-replay )
add_subdirectory( gen-random-bout )