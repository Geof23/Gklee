include( ExternalProject )

set( LLVM_SRC ${CMAKE_SOURCE_DIR}/llvm )

find_program(DOWNLOAD wget)
find_program(EXTRACT tar)
find_program(PATCH patch)
find_program(BASH bash)

set(
  PCMD 
  cd ${LLVM_SRC}/src/LLVM/tools && if [ ! -e clang ]$<SEMICOLON> then ${DOWNLOAD} http://www.llvm.org/releases/3.2/clang-3.2.src.tar.gz && ${EXTRACT} -zxf clang-3.2.src.tar.gz && mv -f clang-3.2.src clang && rm clang-3.2.src.tar.gz && cd clang && cp ${CMAKE_SOURCE_DIR}/patch/clang.patch ./ && ${PATCH} -p1 -N < clang.patch$<SEMICOLON> fi && cd ${LLVM_SRC}/src/LLVM/projects && if [ ! -e compiler-rt ]$<SEMICOLON> then ${DOWNLOAD} http://www.llvm.org/releases/3.2/compiler-rt-3.2.src.tar.gz && ${EXTRACT} -zxf compiler-rt-3.2.src.tar.gz && mv compiler-rt-3.2.src compiler-rt && rm compiler-rt-3.2.src.tar.gz && cd compiler-rt/lib/asan && cp ${LLVM_SRC}/patch/compiler-rt_lib_asan.patch . && ${PATCH} -p1 -N compiler-rt_lib_asan.patch$<SEMICOLON> fi
  )
string(REPLACE ";" " " PCMD "${PCMD}")
ExternalProject_add(
  LLVM
  PREFIX ${LLVM_SRC}
  URL http://www.llvm.org/releases/3.2/llvm-3.2.src.tar.gz
  URL_MD5 71610289bbc819e3e15fdd562809a2d7
  PATCH_COMMAND ${BASH} -c ${PCMD}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} 
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${GLIBCXX} 
  -std=c++11 -DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS} -DLLVM_TARGETS_TO_BUILD=X86
  )

ExternalProject_add(
  libcxx
  DEPENDS LLVM
  PREFIX ${LLVM_SRC}/libcxx
  URL https://releases.llvm.org/3.3/libcxx-3.3.src.tar.gz
  DOWNLOAD_NO_EXTRACT TRUE
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND cd ${LLVM_SRC}/libcxx/src && ${EXTRACT} -f libcxx-3.3.src.tar.gz -x libcxx-3.3.src/include && rm -r libcxx && mv libcxx-3.3.src libcxx
  )

ExternalProject_add(
  STP
  DEPENDS LLVM
  PREFIX ${CMAKE_SOURCE_DIR}/Gklee/STP
  GIT_REPOSITORY https://github.com/stp/stp.git
  GIT_TAG 40b6ca4757b991f1a054c6f9e900ff5e8b3f49db
  PATCH_COMMAND cd ${CMAKE_SOURCE_DIR}/Gklee/STP/src/STP && cp ${CMAKE_SOURCE_DIR}/stp.patch ./ && patch -p1 < stp.patch
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR} -DENABLE_PYTHON_INTERFACE=NO -DENABLE_ASSERTIONS=${ENABLE_ASSERTIONS} -DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}
  )

ExternalProject_add(
  TaintAnalysis
  DEPENDS LLVM
  PREFIX ${LLVM_SRC}/projects/TaintAnalysis
  GIT_REPOSITORY https://github.com/Geof23/TaintAnalysis.git
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}
  
  )

