include ( ${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake )
include ( ${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake )

set ( HAVE_EXT_STP 1 )

set ( HAVE_LIBSTP 1 )

check_function_exists ( __ctype_b_loc HAVE_CTYPE_EXTERNALS )

check_include_files ( inttypes.h HAVE_INTTYPES_H )

check_include_files ( memory.h HAVE_MEMORY_H )

check_include_files ( selinux/selinux.h HAVE_SELINUX_SELINUX_H )

check_include_files ( stdint.h  HAVE_STDINT_H )

check_include_files ( stdlib.h HAVE_STDLIB_H )

check_include_files ( strings.h  HAVE_STRINGS_H )

check_include_files ( string.h HAVE_STRING_H )

check_include_files ( sys/acl.h HAVE_SYS_ACL_H )

check_include_files ( sys/stat.h HAVE_SYS_STAT_H )

check_include_files ( sys/types.h HAVE_SYS_TYPES_H )

check_include_files ( unistd.h  HAVE_UNISTD_H )

set ( KLEE_UCLIBC "${OBJECT_SOURCE_DIR}/klee-uclibc" )

#we'll set the following in SetupExterns.cmake
# LLVM_IS_RELEASE
set ( LLVM_VERSION_MAJOR 3 )
set ( LLVM_VERSION_MINOR 2 )
# set ( LLVM_VERSION_CODE ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} )
# add_definitions( -DLLVM_VERSION_CODE=${LLVM_VERSION_CODE} )

set ( PACKAGE_BUGREPORT "gklee_developers@cs.utah.edu" )

set ( PACKAGE_NAME "GKLEE: concolic verifier-cum-analyzer for CUDA" )

set ( PACKAGE_STRING "${PACKAGE_NAME}, version ${GKLEE_VERSION_MAJOR}.${GKLEE_VERSION_MINOR}" )

set ( PACKAGE_TARNAME "GKLEE_${GKLEE_VERSION_MAJOR}.${GKLEE_VERSION_MINOR}.tgz" )

set ( PACKAGE_VERSION "${GKLEE_VERSION_MAJOR}.${GKLEE_VERSION_MINOR}" )

if ( ENABLE_ASSERTIONS )
  set ( ASSERT "+Asserts" )
else()
  set ( ASSERT "" )
endif()
set ( RUNTIME_CONFIGURATION = "Release${ASSERT}" )
if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
  set ( RUNTIME_CONFIGURATION "Debug${ASSERT}" )
elseif ( ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" )
  set ( RUNTIME_CONFIGURATION "Release+Debug${ASSERT}" )
endif()

check_include_files ( ctype.h STDC_HEADERS )
