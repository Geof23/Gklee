add_executable ( kleaver main.cpp )

add_dependencies( kleaver LLVM STP )

target_link_libraries( kleaver 
  kleaverSolver 
  kleaverExpr 
  kleeSupport 
  kleeBasic
  LLVMSupport
  pthread
  dl
  m				
  stp
)
