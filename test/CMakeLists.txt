function(build_test FNAME)
  add_executable(${FNAME} ${FNAME}.c)
  if("${BM_TARGET}" STREQUAL "soc")
    set(SOC_LIB ${PROTO_LIBRARIES} ${CLABS_LIBRARIES})
  else()
    set(SOC_LIB "")
  endif()
  target_link_libraries(${FNAME} ${SOC_LIB} cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS} -lm)
  install(TARGETS ${FNAME} DESTINATION bin)
endfunction()

if(NOT "${BM_TARGET}" STREQUAL "soc")
  project(test)
  include_directories(
      ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty
      ${CMAKE_CURRENT_SOURCE_DIR}/../include
      ${OpenCV_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/middleware/include
  )

  # Basic op
  add_executable(test_add test_add.cpp)
  target_link_libraries(test_add cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_add DESTINATION bin)
  add_executable(test_and test_and.cpp)
  target_link_libraries(test_and cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_and DESTINATION bin)
  add_executable(test_block test_block.cpp)
  target_link_libraries(test_block cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_block DESTINATION bin)
  add_executable(test_or test_or.cpp)
  target_link_libraries(test_or cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_or DESTINATION bin)
  add_executable(test_sub test_sub.cpp)
  target_link_libraries(test_sub cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_sub DESTINATION bin)
  add_executable(test_xor test_xor.cpp)
  target_link_libraries(test_xor cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_xor DESTINATION bin)

  add_executable(test_filter test_filter.cpp)
  target_link_libraries(test_filter cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_filter DESTINATION bin)
  add_executable(test_morph test_morph.cpp)
  target_link_libraries(test_morph cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_morph DESTINATION bin)
  add_executable(test_threshold test_threshold.cpp)
  target_link_libraries(test_threshold cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_threshold DESTINATION bin)
  add_executable(test_sobel test_sobel.cpp)
  target_link_libraries(test_sobel cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
  install(TARGETS test_sobel DESTINATION bin)
endif()

project(test-c)
include_directories(
    ${CMAKE_SOURCE_DIR}/../3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

build_test(test_add_c)
build_test(test_and_c)
build_test(test_block_c)
build_test(test_copy_c)
build_test(test_hog_c)
build_test(test_morph_c)
build_test(test_norm_grad_c)
build_test(test_filter_c)
build_test(test_or_c)
build_test(test_sad_c)
build_test(test_sobel_grad_c)
build_test(test_sub_c)
build_test(test_threshold_c)
build_test(test_xor_c)