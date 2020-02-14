# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${LIBDEP_BMTAP2_DIR}" STREQUAL "")
    set(LIBDEP_BMTAP2_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BMTAP2_TARGET_BASENAME})
    if(EXISTS "${LIBDEP_BMTAP2_DIR}")
      message("Middleware dolder found.")
    else()
      message("Folder not found, downloading...")
      set(MD5_HASH b60fcf65531c4b71f509700a46173bf4)
      set(FILE_NAME cmodel_bm1880v2-20200214.7z)
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