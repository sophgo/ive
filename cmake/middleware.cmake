# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${LIBDEP_MIDDLEWARE_DIR}" STREQUAL "")
    set(LIBDEP_MIDDLEWARE_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${MIDDLEWARE_TARGET_BASENAME})
    if(EXISTS "${LIBDEP_MIDDLEWARE_DIR}")
      message("Middleware dolder found.")
    else()
      message("Folder not found, downloading...")
      set(MD5_HASH 8d99c5e5de0d2c4a5a61561ba8fe912b)
      set(FILE_NAME middleware-20200213.7z)
      set(WORK_DIR ${CMAKE_SOURCE_DIR}/prebuilt)
      include(${CMAKE_SOURCE_DIR}/cmake/downanddecompress.cmake)
      download_and_decompress(ftp://10.34.33.5/ai/prebuilt/ive/${FILE_NAME}
                              ${FILE_NAME}
                              ${WORK_DIR}
                              MD5
                              ${MD5_HASH})
    endif()
endif()

include_directories(
    ${LIBDEP_MIDDLEWARE_DIR}/include/
)
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "SDKRelease")
  install(DIRECTORY ${LIBDEP_MIDDLEWARE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/middleware)
endif()