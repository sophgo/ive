# Copyright 2018 Bitmain Inc.
# License
# Author Yangwen Huang yangwen.huang@bitmain.com

if(CMAKE_TOOLCHAIN_FILE)
  project(glog)
    set(LIBDEP_SOC_DIR ${CMAKE_SOURCE_DIR}/prebuilt/${BMTAP2_TARGET_BASENAME}/3rdparty_prebuilt)
    include_directories(
        "${LIBDEP_SOC_DIR}/include"
        "${LIBDEP_SOC_DIR}/include/glog"
    )

    set(GLOG_LIBRARIES
        ${LIBDEP_SOC_DIR}/lib/libglog.so
        ${LIBDEP_SOC_DIR}/lib/libglog.so.0
        ${LIBDEP_SOC_DIR}/lib/libglog.so.0.0.0
    )
    install(FILES ${GLOG_LIBRARIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  project(protobuf)
    set(PROTO_LIBRARIES
        ${LIBDEP_SOC_DIR}/lib/libprotobuf.so
        ${LIBDEP_SOC_DIR}/lib/libprotobuf.so.17
        ${LIBDEP_SOC_DIR}/lib/libprotobuf.so.17.0.0
    )
    install(FILES ${PROTO_LIBRARIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  project(cblas)
    set(CLABS_LIBRARIES
        ${LIBDEP_SOC_DIR}/lib/libcblas_LINUX.so
    )
    install(FILES ${CLABS_LIBRARIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
else()
endif()