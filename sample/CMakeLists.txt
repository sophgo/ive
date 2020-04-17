function(buildninstall_test FNAME)
  add_executable(${FNAME} ${FNAME}.c)
  target_link_libraries(${FNAME} cvi_ive_tpu ${GLOG_LIBRARIES} -lm)
  install(TARGETS ${FNAME} DESTINATION bin)
  install(FILES ${FNAME}.c DESTINATION sample)
endfunction()

project(sample)
include_directories(
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${CMAKE_SOURCE_DIR}/include
)

buildninstall_test(sample_add)
buildninstall_test(sample_block)
buildninstall_test(sample_hog)
buildninstall_test(sample_ordstat)