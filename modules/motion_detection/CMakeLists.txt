project(motion_detection)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    ${MIDDLEWARE_INCLUDES})
add_library(${PROJECT_NAME} OBJECT md.cpp)