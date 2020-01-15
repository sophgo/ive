project(ive)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/sqrt.cpp
)

add_library(cvi_tpu_math ${SRC})
target_link_libraries(cvi_tpu_math ${BM_LIBS})
set_target_properties(${FEXT_NAME} PROPERTIES
                     VERSION ${QNN_VERSION_STRING}
                     SOVERSION ${QNN_VERSION_MAJOR})
install(TARGETS cvi_tpu_math DESTINATION lib)