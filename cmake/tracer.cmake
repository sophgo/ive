include(ExternalProject)
# Only Ninja needs this => BUILD_BYPRODUCTS.
ExternalProject_Add(tracer
                    GIT_REPOSITORY ssh://${REPO_USER}10.240.0.84:29418/tracer
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tracer
                    INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    CMAKE_ARGS
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DTOOLCHAIN_ROOT_DIR=${TOOLCHAIN_ROOT_DIR}
                    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                    -DCMAKE_INSTALL_MESSAGE=LAZY
                    -DBUILD_SHARED_LIBS=ON
                    -DCMAKE_CXX_FLAGS="-fPIC")
ExternalProject_Get_Property(tracer INSTALL_DIR)
message("Content downloaded to ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tracer")
if (ENABLE_SYSTRACE)
install(DIRECTORY ${INSTALL_DIR}/ DESTINATION . USE_SOURCE_PERMISSIONS)
set(TRACER_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
endif()