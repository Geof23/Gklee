include( ExternalProject )

set( LLVM_SRC ${CMAKE_SOURCE_DIR}/llvm )

message( "We made it into SetupExterns.cmake" )

ExternalProject_add(
   LLVM
   PREFIX ${LLVM_SRC}
   URL http://llvm.org/releases/3.6.0/llvm-3.6.0.src.tar.xz
   #URL_MD5 71610289bbc819e3e15fdd562809a2d7
   #we're going to abuse PATCH_COMMAND by adding Clang (along with its patch), compiler-rt and TaintAnalysis to the llvm tree here
   PATCH_COMMAND cd ${LLVM_SRC}/src/LLVM/tools && wget http://llvm.org/releases/3.6.0/cfe-3.6.0.src.tar.xz && tar -Jxf cfe-3.6.0.src.tar.xz && mv cfe-3.6.0.src clang && rm cfe-3.6.0.src.tar.xz && cd clang && cd ${LLVM_SRC}/src/LLVM/projects && wget http://llvm.org/releases/3.6.0/compiler-rt-3.6.0.src.tar.xz && tar -Jxf compiler-rt-3.6.0.src.tar.xz && mv compiler-rt-3.6.0.src compiler-rt && rm compiler-rt-3.6.0.src.tar.xz 
   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}
)

#cp ${CMAKE_SOURCE_DIR}/clang.patch ./ && patch -p1 < clang.patch &&  -- llvm 3.6 has better support for CUDA, and there are conflicting names in the patch with the newer Clang attributes.  Let's take a look at what we get when we compile with clang++ '... -x cuda -Xclang -fgpu-device'?

#http://www.llvm.org/releases/3.2/clang-3.2.src.tar.gz http://www.llvm.org/releases/3.2/compiler-rt-3.2.src.tar.gz  #http://www.llvm.org/releases/3.2/llvm-3.2.src.tar.gz

# ExternalProject_add(
#   COMPILER-RT
#   DEPENDS LLVM
#   PREFIX ${LLVM_SRC}/projects/compiler-rt
#   URL http://www.llvm.org/releases/3.2/compiler-rt-3.2.src.tar.gz
#   # SOURCE_DIR ${LLVM_SRC}/projects/compiler-rt
#   # BINARY_DIR ${LLVM_SRC}/projects/compiler-rt
# 	CMAKE_ARGS -DCMAKE_MODULE_PATH=${LLVM_SRC}/src/LLVM/cmake/modules
#  )

# ExternalProject_add(
#   CLANG
#   DEPENDS LLVM COMPILER-RT
#   PREFIX ${LLVM_SRC}/tools/clang
#   URL http://www.llvm.org/releases/3.2/clang-3.2.src.tar.gz
#   PATCH_COMMAND cd ${LLVM_SRC}/tools/clang/src/CLANG && cp ../../../../../clang.patch ./ && patch -p1 < clang.patch
#   # SOURCE_DIR ${LLVM_SRC}/tools/clang
#   # BINARY_DIR ${LLVM_SRC}/tools/clang
#   CMAKE_ARGS -DCLANG_PATH_TO_LLVM_BUILD=${LLVM_SRC}/src/LLVM-build -DCMAKE_MODULE_PATH=${LLVM_SRC}/src/LLVM/cmake/modules
#  )

ExternalProject_add(
  STP
  DEPENDS LLVM
  PREFIX ${CMAKE_SOURCE_DIR}/Gklee/STP
  GIT_REPOSITORY https://github.com/stp/stp.git
  GIT_TAG 40b6ca4757b991f1a054c6f9e900ff5e8b3f49db
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/Gklee/stp -DENABLE_PYTHON_INTERFACE=NO -DENABLE_ASSERTIONS=${ENABLE_ASSERTIONS}
)

ExternalProject_add(
  TaintAnalysis
  DEPENDS LLVM
  PREFIX ${LLVM_SRC}/projects/TaintAnalysis
  GIT_REPOSITORY https://github.com/Geof23/TaintAnalysis.git
	#TODO restore tag to correct snapshot
#  GIT_TAG 4755611618c1c539f03dd6629503a0167f137dd7
   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}
)