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

if( DO_GLIBCXX_DEBUG )
  set( GLIBCXX -DGLIBCXX_DEBUG )
endif()

# Search paths for custom CMake modules
set( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules )
#set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${LLVM_SRC}/src/LLVM/cmake/modules )


#does system check for basic capabilities (for settings in config.h.in, including e.g. PACKAGE_BUGREPORT)
include( ${CMAKE_MODULE_PATH}/Introspect.cmake )

CONFIGURE_FILE( Gklee/include/klee/Config/config.h.in Gklee/include/klee/Config/config.h )

find_package(Doxygen)
if(DOXYGEN_FOUND)
  message( "Found Doxygen" )
  CONFIGURE_FILE( doc/Doxyfile.in doc/Doxyfile )
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
else()
  message( "No Doxygen found" )
endif(DOXYGEN_FOUND)


#installs external projects -- stp and llvm
MESSAGE( "loading SetupExterns.cmake" )
include( ${CMAKE_MODULE_PATH}/SetupExterns.cmake )


message( "CMAKE_MODULE_PATH is ${CMAKE_MODULE_PATH}" )

# -----------------------------------------------------------------------------
# Set the appropriate build flags
# -----------------------------------------------------------------------------
include( ${CMAKE_MODULE_PATH}/CompilerFlags.cmake )

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

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)


add_subdirectory(Gklee)

