message( "we entered BuildBCArch.cmake . . ." )

set( BC_FLAGS -I${HOME}/Gklee/include -I${HOME}/Gklee/runtime/Intrinsic -I${HOME}/llvm/src/LLVM/include -I${BINARIES}/Gklee/include -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -D__NO_INLINE__ -g -fPIC -std=gnu89 -g -O2 -Wall -W -Wno-unused-parameter -Wwrite-strings )

file( GLOB files ${SOURCE}/*.c )

foreach (SF ${files} )
    message( "clang command: ${BBBIN}/clang ${BC_FLAGS} ${SF} -o ${SF}.ll -S -emit-llvm" )
    execute_process( COMMAND ${BBBIN}/clang ${BC_FLAGS} ${SF} -o ${SF}.ll -S -emit-llvm
      OUTPUT_VARIABLE OUT
      ERROR_VARIABLE OUT
      )
    message( "output from clang is ${OUT}" )
    message( "llvm-as command: ${BBBIN}/llvm-as ${SF}.ll -o - | ${BBBIN}/opt -std-compile-opts -o ${SF}.bc" )
    execute_process( COMMAND ${BBBIN}/llvm-as ${SF}.ll -o - 
      COMMAND ${BBBIN}/opt -std-compile-opts -o ${SF}.bc
      OUTPUT_VARIABLE OUT
      ERROR_VARIABLE OUT
      )
    message( "output from llvm-as is ${OUT}" )
endforeach(SF)

message( "building ${DEST}" )
file( REMOVE ${DEST} )
file( GLOB BCS ${SOURCE}/*.c.bc )
message( "building ${DEST} with ${BBBIN}/llvm-ar rcsf ${DEST} ${BCS}" )
execute_process( COMMAND ${BBBIN}/llvm-ar rcsf ${DEST} ${BCS}
      OUTPUT_VARIABLE OUT
      ERROR_VARIABLE OUT
      )
message( "output from llvm-ar is ${OUT}" )  

# set( LLVM_BIN ${CMAKE_SOURCE_DIR}/llvm/src/LLVM-build/bin )
# set( CLANG_BIN ${CMAKE_SOURCE_DIR}/llvm/tools/clang/src/CLANG-build/bin )

# function( BUILD_BYTECODE_AR FLAGS FILES DESTINATION TEMP_DIR )
  # message( "in function BUILD_BYTECODE_AR:" )
  # message( "FLAGS is ${FLAGS}" )
  # message( "FILES is ${FILES}" )
  # message( "DESTINATION is ${DEST}" )
  # message( "BBBIN is ${BBBIN}" )
#  message( "TEMP_DIR is ${TEMP_DIR}" )
# message( "Let's glob the contents of the pwd . . ." )
# file( GLOB LIST * )
# message( "this is what the glob list looks like: ${LIST}")
# message( "now individually. . . ." )
# foreach( F ${LIST} )
#   message("F is ${F}" )
# endforeach(F)

#   foreach( SF ${FILES} )
#     message( "clang command: ${BBBIN}/clang++ ${FLAGS} ${SF} -o ${SF}.ll -S -emit-llvm" )
#     execute_process( COMMAND "${BBBIN}/clang++ ${FLAGS} ${SF} -o ${SF}.ll -S -emit-llvm"
#       OUTPUT_FILE bst_clang_out_${SF}
#       )
#     message( "llvm-as command: ${BBBIN}/llvm-as ${SF}.ll -o - | ${BBBIN}/opt -std-compile-opts -o ${SF}.bc" )
#     execute_process( COMMAND "${BBBIN}/llvm-as ${SF}.ll -o - | ${BBBIN}/opt -std-compile-opts -o ${SF}.bc"
#       OUTPUT_FILE bst_llvm-as_out${SF}
# )
#   endforeach( SF )
#   file( GLOB BC_FILES *.bc )
#   execute_process( COMMAND "${BBBIN}/llvm-ar rcsf ${DEST} ${BC_FILES}" )
  #install( TARGETS ${DEST} DESTINATION ${CMAKE_SOURCE_DIR} )
# endfunction( BUILD_BYTECODE_AR )

message( "we exited BuildBCArch.cmake . . ." )