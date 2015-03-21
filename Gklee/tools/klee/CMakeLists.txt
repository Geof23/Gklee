set( CMAKE_INCLUDE_CURRENT_DIR ON )

add_executable ( klee Debug.cpp main.cpp )

add_dependencies( klee LLVM STP )

set ( LIBS kleeCore kleeModule kleaverSolver
  kleaverExpr kleeSupport kleeBasic
  LLVMX86AsmParser LLVMX86Disassembler
  LLVMX86CodeGen LLVMSelectionDAG
  LLVMAsmPrinter LLVMMCParser
  LLVMX86Desc LLVMX86Info
  LLVMX86AsmPrinter LLVMX86Utils LLVMLinker
  LLVMArchive LLVMipo
  LLVMVectorize LLVMBitWriter LLVMBitReader
  LLVMJIT LLVMRuntimeDyld LLVMExecutionEngine
  LLVMCodeGen LLVMScalarOpts LLVMInstCombine
  LLVMTransformUtils LLVMipa
  LLVMAnalysis LLVMTarget LLVMMC LLVMObject
  LLVMCore LLVMSupport pthread dl m stp
)
										

target_link_libraries( klee ${LIBS} )
