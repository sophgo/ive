# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${LIBDEP_BMTAP2_DIR}" STREQUAL "")
    set(LIBDEP_BMTAP2_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BMTAP2_TARGET_BASENAME})
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