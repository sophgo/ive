function(build_test FNAME)
  add_executable(${FNAME} ${FNAME}.c)
  if("${CVI_TARGET}" STREQUAL "soc")
    set(SOC_LIB ${PROTO_LIBRARIES} ${CLABS_LIBRARIES})
  else()
    set(SOC_LIB "")
  endif()
  target_link_libraries(${FNAME} ${SOC_LIB} cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS} -lm)
  install(TARGETS ${FNAME} DESTINATION bin)
endfunction()

project(sample)
include_directories(
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${CMAKE_SOURCE_DIR}/include
)

build_test(sample_add)