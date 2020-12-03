cmake_minimum_required(VERSION 3.2.2)

# Options
option(ENABLE_SYSTRACE "Use systrace" OFF)
execute_process(COMMAND git describe OUTPUT_VARIABLE LATEST_TAG)
if ("${LATEST_TAG}" STREQUAL "")
  message(WARNING "Cannot fetch latest tag from git.")
  set(LATEST_TAG "v0.0.0")
endif()
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)*[^(\\\n).]+" LATEST_TAG ${LATEST_TAG})
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" MAJOR_VERSION ${LATEST_TAG})
string(REGEX MATCH "^([^.]+)" SO_VERSION ${MAJOR_VERSION})

if (CMAKE_C_COMPILER STREQUAL "/usr/bin/cc")
  message(FATAL_ERROR "Please use clang compiler. Please export C and CXX compiler to clang.")
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if("${CVI_PLATFORM}" STREQUAL "")
  set(CVI_PLATFORM "cv1835")
  message(AUTHOR_WARNING "Platform not provided, set to ${CVI_PLATFORM}.")
endif()

# Reset flag string
set(WORKAROUND_FLAGS_STR "")

# platform specific settings
if("${CVI_PLATFORM}" STREQUAL "cv1835")
  add_definitions(-DWORKAROUND_SCALAR_4096_ALIGN_BUG)
  set(WORKAROUND_FLAGS_STR "${WORKAROUND_FLAGS_STR} WORKAROUND_SCALAR_4096_ALIGN_BUG")
endif()

if("${CVI_TARGET}" STREQUAL "soc")
  add_definitions(-DCVI_SOC)
  if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install_soc")
  endif()
  # Find toolchain cmake file and toolchain folder
  if("${CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
    message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is not set. Aborting.")
  endif()
  if(NOT DEFINED TOOLCHAIN_ROOT_DIR)
    message(FATAL_ERROR "TOOLCHAIN_ROOT_DIR is not set. Aborting.")
  endif()
  set(USE_TOOLCHAIN "true")
  set(SSE_FLAGS "")
  set(GCC_STRIP_FLAG "-s")
else()
  set(CVI_TARGET "cmodel")
  if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")
  endif()
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(USE_TOOLCHAIN "false")
  set(SSE_FLAGS "-mssse3 -msse4.1 -mavx2 -mfma")
  set(GCC_STRIP_FLAG "")
endif()

# Check if tracer exists.
if ("${TRACER_ROOT}" STREQUAL "")
  set(TRACER_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/tracer)
endif()
if(EXISTS "${TRACER_ROOT}")
  execute_process(COMMAND git remote -v
                  WORKING_DIRECTORY ${TRACER_ROOT}
                  OUTPUT_VARIABLE TRACER_REMOTE_URL)
  if (NOT "${TRACER_REMOTE_URL}" MATCHES "10.34.33.3:8480" OR
          "${TRACER_REMOTE_URL}" MATCHES "10.34.33.3:8422")
    message(AUTHOR_WARNING "Warning, this tracer is not cloned from GitLab. "
                           "May cause error while compiling or running.")
  endif()
  message("-- Found Tracer (directory: ${CMAKE_SOURCE_DIR}/3rdparty/tracer)")
  include(cmake/tracer.cmake)
else()
  message(FATAL_ERROR "Tracer folder not found. Git clone to 3rdparty at "
                      "http://10.34.33.3:8480/sys_app/tracer.")
endif()
if (ENABLE_SYSTRACE)
  add_definitions(-DENABLE_TRACE)
endif()

# Check if MLIR SDK exists.
include(cmake/mlir.cmake)

# Check if middleware exists.
include (cmake/middleware.cmake)

#include sysroot/include
include_directories(${CMAKE_SYSROOT}/include)

# include kernel headers for user space
include_directories(${KERNEL_HEADERS_ROOT}/include)

set(CMAKE_C_INIT "-fsigned-char -Werror=all -fdiagnostics-color=always ${SSE_FLAGS}")
set(CMAKE_CXX_INIT "-fsigned-char -fPIC -Werror=all -fdiagnostics-color=always -std=gnu++11 ${SSE_FLAGS}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} ${CMAKE_C_FLAGS_RELEASE}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} ${CMAKE_CXX_FLAGS_RELEASE}" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} ${CMAKE_C_FLAGS_RELEASE} ${GCC_STRIP_FLAG}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} ${CMAKE_CXX_FLAGS_RELEASE} ${GCC_STRIP_FLAG}" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} ${CMAKE_C_FLAGS_DEBUG}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} ${CMAKE_CXX_FLAGS_DEBUG}" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Asan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} ${CMAKE_C_FLAGS_DEBUG} -fsanitize=address" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Tsan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} ${CMAKE_C_FLAGS_DEBUG} -fsanitize=thread" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread" )
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "UBsan")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_INIT} ${CMAKE_C_FLAGS_DEBUG} -fsanitize=undefined" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_INIT} ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined" )
    set( CMAKE_EXE_LINKER_FLAGS "-lubsan")
else()
    message(FATAL_ERROR "No build type!!!")
endif()

# Get current commit
execute_process(COMMAND git show --oneline -s OUTPUT_VARIABLE CURRENT_COMMIT)

message("==================================================")
message("Latest tag       ${LATEST_TAG}")
message("Commit           ${CURRENT_COMMIT}")
message("[Summary]")
message("Platform         ${CVI_PLATFORM}")
message("Target           ${CVI_TARGET}")
message("Build type       ${CMAKE_BUILD_TYPE}")
message("C   compiler     ${CMAKE_C_COMPILER}")
message("CXX compiler     ${CMAKE_CXX_COMPILER}")
message("C Flags          ${CMAKE_C_FLAGS}")
message("CXX Flags        ${CMAKE_CXX_FLAGS}")
message("SYSROOT          ${CMAKE_SYSROOT}")
message("Install dir      ${CMAKE_INSTALL_PREFIX}")
message("Use Systrace     ${ENABLE_SYSTRACE}")
message("")
message("Workaround flags ${WORKAROUND_FLAGS_STR}")
message("==================================================")

# add_subdirectory(tpu_math)
add_subdirectory(src)
add_subdirectory(sample)
#if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  add_subdirectory(test)
#endif()
# Install cmake file
install(FILES cmake/ive.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake/)
