# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>

if("${MLIR_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "You must set MLIR_SDK_ROOT before building IVE library.")
elseif(EXISTS "${MLIR_SDK_ROOT}")
  message("-- Found MLIR_SDK_ROOT (directory: ${MLIR_SDK_ROOT})")
else()
  message(FATAL_ERROR "${MLIR_SDK_ROOT} is not a valid folder.")
endif()

project(mlir-sdk)
set(MLIR_INCLUDES
    ${MLIR_SDK_ROOT}/include/
)
set(MLIR_LIBS
    ${MLIR_SDK_ROOT}/lib/libcvikernel.so
    ${MLIR_SDK_ROOT}/lib/libcviruntime.so
)

if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  install(DIRECTORY ${MLIR_SDK_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
  install(FILES ${MLIR_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()

if(CMAKE_TOOLCHAIN_FILE)
  project(glog)
  set(GLOG_LIBRARIES
      ${MLIR_SDK_ROOT}/lib/libglog.so
      ${MLIR_SDK_ROOT}/lib/libglog.so.0
      ${MLIR_SDK_ROOT}/lib/libglog.so.0.0
      ${MLIR_SDK_ROOT}/lib/libglog.so.0.0.0
  )

  if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
    install(FILES ${GLOG_LIBRARIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  endif()
endif()