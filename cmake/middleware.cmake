# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang <yangwen.huang@bitmain.com>


if("${LIBDEP_MIDDLEWARE_DIR}" STREQUAL "")
    set(LIBDEP_MIDDLEWARE_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${MIDDLEWARE_TARGET_BASENAME})
endif()

include_directories(
    ${LIBDEP_MIDDLEWARE_DIR}/include/
)

install(DIRECTORY ${LIBDEP_MIDDLEWARE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)