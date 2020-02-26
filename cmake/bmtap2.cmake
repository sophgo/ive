# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${LIBDEP_BMTAP2_DIR}" STREQUAL "")
    set(LIBDEP_BMTAP2_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BMTAP2_TARGET_BASENAME})
    if(EXISTS "${LIBDEP_BMTAP2_DIR}")
      message("Middleware dolder found.")
    else()
      message("Folder not found, downloading...")
      if ("${BMTAP2_TARGET_BASENAME}" STREQUAL "cmodel_bm1880v2")
        set(MD5_HASH 453c5fcf9670f0e5b53e8519b08e0c7f)
        set(DATE_TAG 20200226)
      elseif ("${BMTAP2_TARGET_BASENAME}" STREQUAL "soc_bm1880v2")
        set(MD5_HASH 23e7d7943b53d56004c58040ef222a43)
        set(DATE_TAG 20200226)
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
install(DIRECTORY ${LIBDEP_BMTAP2_DIR}/lib/ DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)