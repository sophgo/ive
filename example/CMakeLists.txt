project(example)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/middleware/include
)

add_executable(example_bk_subtract example_bk_subtract.cpp)
target_link_libraries(example_bk_subtract cvi_ive_tpu ${GLOG_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS example_bk_subtract DESTINATION bin)
