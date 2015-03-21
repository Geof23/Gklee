set( CMAKE_INCLUDE_CURRENT_DIR ON )

file( GLOB SOURCES *.c )

add_executable ( klee-replay ${SOURCES} )

add_dependencies ( klee-replay LLVM )

target_link_libraries( klee-replay kleeBasic pthread dl m util)

