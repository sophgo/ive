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
buildninstall_test(sample_and)
buildninstall_test(sample_block)
buildninstall_test(sample_copy)
buildninstall_test(sample_hog)
buildninstall_test(sample_magandang)
buildninstall_test(sample_map)
buildninstall_test(sample_morph)
buildninstall_test(sample_normgrad)
buildninstall_test(sample_or)
buildninstall_test(sample_ordstat)
buildninstall_test(sample_sub)
buildninstall_test(sample_xor)