# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${LIBDEP_BMTAP2_DIR}" STREQUAL "")
    set(LIBDEP_BMTAP2_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BMTAP2_TARGET_BASENAME})
    if(EXISTS "${LIBDEP_BMTAP2_DIR}")
      message("Bmtap2 dolder found.")
    else()
      message("Folder not found, downloading...")
      if ("${BMTAP2_TARGET_BASENAME}" STREQUAL "cmodel_bm1880v2")
        set(MD5_HASH 42ff5f3800e694ac432675a67531fd41)
        set(DATE_TAG 20200326)
      elseif ("${BMTAP2_TARGET_BASENAME}" STREQUAL "soc_bm1880v2")
        set(MD5_HASH 3e53f08673116a133a70d7b33828b944)
        set(DATE_TAG 20200326)
      else()
        message(FATAL_ERROR "Unknown bmtap2 target basename.")
      endif()
      set(FILE_NAME ${BMTAP2_TARGET_BASENAME}-${DATE_TAG}.7z)
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
    ${LIBDEP_BMTAP2_DIR}/include/
)

set(BM_LIBS
    ${LIBDEP_BMTAP2_DIR}/lib/libbmodel.so
    ${LIBDEP_BMTAP2_DIR}/lib/libbmkernel.so
    ${LIBDEP_BMTAP2_DIR}/lib/libbmruntime.so
    ${LIBDEP_BMTAP2_DIR}/lib/libbmutils.so
    #${LIBDEP_BMTAP2_DIR}/lib/libclas_LINUX.so
)

install(DIRECTORY ${LIBDEP_BMTAP2_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
install(FILES ${BM_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)