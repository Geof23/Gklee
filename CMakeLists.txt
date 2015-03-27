project( GKLEE )
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set ( GKLEE_VERSION_MAJOR 0)
set ( GKLEE_VERSION_MINOR 1)

set( CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} )

set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
# -----------------------------------------------------------------------------
# Make RelWithDebInfo the default build type if otherwise not set
# -----------------------------------------------------------------------------
set(build_types Debug Release RelWithDebInfo MinSizeRel)
if(NOT CMAKE_BUILD_TYPE)

      message(STATUS "You can choose the type of build, options are:${build_types}")
      set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE String
          "Options are ${build_types}"
          FORCE
         )

      # Provide drop down menu options in cmake-gui
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${build_types})
endif()
message(STATUS "Doing a ${CMAKE_BUILD_TYPE} build")

# -----------------------------------------------------------------------------
# Option to enable/disable assertions
# -----------------------------------------------------------------------------

# Filter out definition of NDEBUG from the default build configuration flags.
# We will add this ourselves if we want to disable assertions
foreach (build_config ${build_types})
    string(TOUPPER ${build_config} upper_case_build_config)
    foreach (language CXX C)
        set(VAR_TO_MODIFY "CMAKE_${language}_FLAGS_${upper_case_build_config}")
        string(REGEX REPLACE "(^| )[/-]D *NDEBUG($| )"
                             " "
                             replacement
                             "${${VAR_TO_MODIFY}}"
              )
        #message("Original (${VAR_TO_MODIFY}) is ${${VAR_TO_MODIFY}} replacement is ${replacement}")
        set(${VAR_TO_MODIFY} "${replacement}" CACHE STRING "Default flags for ${build_config} configuration" FORCE)
    endforeach()
endforeach()

option(ENABLE_ASSERTIONS "Build with assertions enabled" ON)
if (ENABLE_ASSERTIONS)
    # NDEBUG was already removed.
else()
    # Note this definition doesn't appear in the cache variables.
    add_definitions( "-DNDEBUG" )
endif()

# Search paths for custom CMake modules
set( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules )
#set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${LLVM_SRC}/src/LLVM/cmake/modules )

include( ${CMAKE_MODULE_PATH}/CompilerFlags.cmake )


#does system check for basic capabilities (for settings in config.h.in, including e.g. PACKAGE_BUGREPORT)
include( ${CMAKE_MODULE_PATH}/Introspect.cmake )

CONFIGURE_FILE( Gklee/include/klee/Config/config.h.in Gklee/include/klee/Config/config.h)


#installs external projects -- stp and llvm
MESSAGE( "loading SetupExterns.cmake" )
include( ${CMAKE_MODULE_PATH}/SetupExterns.cmake )


message( "CMAKE_MODULE_PATH is ${CMAKE_MODULE_PATH}" )

# -----------------------------------------------------------------------------
# Set the appropriate build flags
# -----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

macro(add_cxx_flag_if_supported flagname)
	string( SUBSTRING ${flagname} 1 -1 newflag )
	message( "newflag is ${newflag}" )
  check_cxx_compiler_flag("${flagname}" HAVE_FLAG_${newflag})

  if(HAVE_FLAG_${newflag})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flagname}")
  endif()
endmacro()

if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared library currently broken due to mix of C++/C code")
    add_cxx_flag_if_supported("-fPIC")
endif()

add_cxx_flag_if_supported("-std=c++11")
add_cxx_flag_if_supported("-Wall")
#add_cxx_flag_if_supported("-Wextra")
add_cxx_flag_if_supported("-pedantic")
add_cxx_flag_if_supported("-Wunused")
add_cxx_flag_if_supported("-Wsign-compare")
add_cxx_flag_if_supported("-Wtype-limits")
add_cxx_flag_if_supported("-Wuninitialized")
add_cxx_flag_if_supported("-Wno-deprecated")
add_cxx_flag_if_supported("-Wstrict-aliasing")
add_cxx_flag_if_supported("-Wpointer-arith")
add_cxx_flag_if_supported("-Wheader-guard")
add_cxx_flag_if_supported("-fno-exceptions")
add_cxx_flag_if_supported("-fPIC")
add_cxx_flag_if_supported("-Woverloaded-virtual")
add_cxx_flag_if_supported("-Wcast-qual")
add_cxx_flag_if_supported("-pedantic")
add_cxx_flag_if_supported("-fno-rtti")
add_cxx_flag_if_supported("-Wno-unused_parameter")
add_cxx_flag_if_supported("-Wwrite-strings")


add_definitions("-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -D_GNU_SOURCE")

option(TUNE_NATIVE "Use -mtune=native" OFF)
if(TUNE_NATIVE)
  add_cxx_flag_if_supported("-mtune=native")
endif()


message( "done checking flags.  They're now: ${CMAKE_CXX_FLAGS}" )

#-----------------------------------------------------------------------------
#Enable LLVM sanitizations.
#Note that check_cxx_compiler_flag doesn't work, a fix is needed here
#-----------------------------------------------------------------------------
macro(add_cxx_flag flagname)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flagname}")
endmacro()

option(SANITIZE "Use Clang sanitizers. This will force using clang++ as the compiler" OFF)
if (SANITIZE)
    # Set in Cache so user can tweak it later
    SET(CMAKE_CXX_COMPILER "clang++" CACHE FILEPATH "" FORCE)
    message("Forcing compiler:${CMAKE_CXX_COMPILER}")
    add_cxx_flag("-fsanitize=return")
    add_cxx_flag("-fsanitize=bounds")
    add_cxx_flag("-fsanitize=integer")
    add_cxx_flag("-fsanitize=undefined")
    add_cxx_flag("-fsanitize=float-divide-by-zero")
    add_cxx_flag("-fsanitize=integer-divide-by-zero")
    add_cxx_flag("-fsanitize=null")
    add_cxx_flag("-fsanitize=unsigned-integer-overflow")
    add_cxx_flag("-fsanitize=address")
    add_cxx_flag("-Wno-bitfield-constant-conversion")
endif()

include(CheckCXXCompilerFlag)

macro(add_cxx_flag flagname)
  check_cxx_compiler_flag("${flagname}" HAVE_FLAG_${flagname})

  if(HAVE_FLAG_${flagname})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flagname}")
  endif()
endmacro()


add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)


add_subdirectory(Gklee)

