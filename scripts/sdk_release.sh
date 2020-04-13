#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IVE_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$IVE_ROOT/tmp
# Clone 3rdparty library systrace
pushd $IVE_ROOT/3rdparty
if [ ! -d "tracer" ]; then
  git clone ssh://git@10.34.33.3:8422/sys_app/tracer.git
fi
pushd tracer
git checkout origin/master
popd
popd

# IVE_SDK_NAME="ivesdk"-$(date '+%Y%m%d')
echo "Creating tmp working directory."
if [[ "$1" == "cmodel" ]]; then
    mkdir -p $TMP_WORKING_DIR/build_cmodel
    pushd $TMP_WORKING_DIR/build_cmodel
    cmake -G Ninja $IVE_ROOT -DENABLE_SYSTRACE=OFF \
                             -DCMAKE_BUILD_TYPE=SDKRelease \
                             -DMLIR_SDK_ROOT=$TPU_MLIR_INSTALL_PATH \
                             -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                             -DCMAKE_INSTALL_PREFIX=$IVE_CMODEL_INSTALL_PATH \
    ninja -j8 && ninja install
    popd
elif [[ "$1" == "soc" ]]; then
    mkdir -p $TMP_WORKING_DIR/build_sdk
    pushd $TMP_WORKING_DIR/build_sdk
    cmake -G Ninja $IVE_ROOT -DCVI_TARGET=soc \
                             -DENABLE_SYSTRACE=OFF \
                             -DCMAKE_BUILD_TYPE=SDKRelease \
                             -DMLIR_SDK_ROOT=$TPU_SDK_INSTALL_PATH \
                             -DMIDDLEWARE_SDK_ROOT=$MW_PATH \
                             -DCMAKE_INSTALL_PREFIX=$IVE_SDK_INSTALL_PATH \
                             -DTOOLCHAIN_ROOT_DIR=$HOST_TOOL_PATH \
                             -DCMAKE_TOOLCHAIN_FILE=$IVE_ROOT/toolchain/toolchain-aarch64-linux.cmake
    ninja -j8 && ninja install
    popd
else
  echo "Unsupported build type."
  exit 1
fi
# cd $TMP_WORKING_DIR
# echo "Compressing SDK release..."
# tar cf - ivesdk -P | pv -s $(du -sb ivesdk | awk '{print $1}') | gzip > $IVE_ROOT/$IVE_SDK_NAME.tar.gz
# echo "Output md5 sum."
# md5sum $IVE_ROOT/$IVE_SDK_NAME.tar.gz
echo "Cleanup tmp folder."
rm -r $TMP_WORKING_DIR
