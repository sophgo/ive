project(ive)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_and.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_morph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_or.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_sobel_gradonly.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_sobel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_sub_abs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_sub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_threshold_high_low.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_threshold_slope.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_threshold.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_xor.cpp
)

add_library(cvi_ive_tpu SHARED ${SRC})
target_link_libraries(cvi_ive_tpu cvi_tpu_math ${BM_LIBS})
set_target_properties(${FEXT_NAME} PROPERTIES
                     VERSION ${QNN_VERSION_STRING}
                     SOVERSION ${QNN_VERSION_MAJOR})
install(TARGETS cvi_ive_tpu DESTINATION lib)