#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IVE_ROOT=$(readlink -f $SCRIPT_DIR/../)
TMP_WORKING_DIR=$IVE_ROOT/tmp
IVE_SDK_NAME="ivesdk"-$(date '+%Y%m%d')
echo "Creating tmp working directory."
mkdir -p $TMP_WORKING_DIR/build_sdk
cd $TMP_WORKING_DIR/build_sdk
cmake $IVE_ROOT -DBM_TARGET=soc \
                -DENABLE_SYSTRACE=OFF \
                -DCMAKE_BUILD_TYPE=SDKRelease \
                -DCMAKE_INSTALL_PREFIX=$TMP_WORKING_DIR/ivesdk \
                -DTOOLCHAIN_ROOT_DIR=$IVE_ROOT/../gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu \
                -DCMAKE_TOOLCHAIN_FILE=$IVE_ROOT/toolchain/toolchain-aarch64-linux.cmake
make -j8 && make install
cd $TMP_WORKING_DIR
echo "Compressing SDK release..."
tar cf - ivesdk -P | pv -s $(du -sb ivesdk | awk '{print $1}') | gzip > $IVE_ROOT/$IVE_SDK_NAME.tar.gz
echo "Output md5 sum."
md5sum $IVE_ROOT/$IVE_SDK_NAME.tar.gz
echo "Cleanup tmp folder."
rm -r $TMP_WORKING_DIR
