#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IVE_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$IVE_ROOT/tmp

# IVE_SDK_NAME="ivesdk"-$(date '+%Y%m%d')
echo "Creating tmp working directory."
mkdir -p $TMP_WORKING_DIR/build_sdk
pushd $TMP_WORKING_DIR/build_sdk
cmake -G Ninja $IVE_ROOT -DCVI_TARGET=soc \
                         -DENABLE_SYSTRACE=OFF \
                         -DCMAKE_BUILD_TYPE=SDKRelease \
                         -DKERNEL_HEADERS_ROOT=$KERNEL_HEADER_PATH \
                         -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                         -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                         -DCMAKE_INSTALL_PREFIX=$IVE_SDK_INSTALL_PATH \
                         -DCMAKE_TOOLCHAIN_FILE=$IVE_ROOT/toolchain/toolchain.cmake
ninja -j8 && ninja install
popd
echo "Cleanup tmp folder."
rm -r $TMP_WORKING_DIR
