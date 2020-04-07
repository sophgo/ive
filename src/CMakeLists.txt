project(ive)

include_directories(
    ${MLIR_INCLUDES}
    ${MIDDLEWARE_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../prebuilt/middleware/include
)

if (ENABLE_SYSTRACE)
  set(IVE_TRACER ${CMAKE_CURRENT_SOURCE_DIR}/ive_tracer.cpp)
else()
  set(IVE_TRACER)
endif()

set(SRC
    ${IVE_TRACER}
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/tracer/tracer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/table_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_and.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_block_bf16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_copy_direct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_copy_interval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_magandang.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_morph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_normalize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_or.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_sad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tpu/tpu_sigmoid.cpp
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
target_link_libraries(cvi_ive_tpu ${MLIR_LIBS})
set_target_properties(cvi_ive_tpu PROPERTIES
                     VERSION ${CVI_IVE_MAJOR_VERSION}
                     SOVERSION ${CVI_IVE_SO_VERSION})
add_library(cvi_ive_tpu-static ${SRC})
target_link_libraries(cvi_ive_tpu-static ${MLIR_LIBS})
install(TARGETS cvi_ive_tpu DESTINATION lib)
install(TARGETS cvi_ive_tpu-static DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/cvi_comm_ive.h DESTINATION include/ive)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/ive.h DESTINATION include/ive)
if (ENABLE_SYSTRACE)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/ive_tracer.h DESTINATION include/ive)
endif()