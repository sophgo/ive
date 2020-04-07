#!/bin/sh

gstreplace=gstcviivebackground
pkg=gstreamer-video-1.0

BUILD_TYPE=$1
BASEDIR=$(dirname $0)
CMAKE_SOURCE_DIR=${BASEDIR}
if [ "$BUILD_TYPE" = "x86" ]
then
  IVE_INSTALL_ROOT=${CMAKE_SOURCE_DIR}/../../install
else
  IVE_INSTALL_ROOT=
  TOOLCHAIN_ROOT=
  GST_ROOT_FOLDER=
fi

PROJECT_CFLAGS="-I${IVE_INSTALL_ROOT}/include \
-I${IVE_INSTALL_ROOT}/include/ive
-I${IVE_INSTALL_ROOT}/include/middleware
"

SRC_FILES="\
"

helpFunction()
{
   echo ""
   echo "Usage: $0 type"
   echo "\t-t x86 / edb"
   exit 1
}

if [ -z "$BUILD_TYPE" ]
then
    helpFunction
fi

rm -f *.o *.so

if [ "$BUILD_TYPE" = "x86" ]
then
    gcc -O3 -Wall -fPIC $CPPFLAGS $(pkg-config --cflags gstreamer-1.0 $pkg opencv) ${PROJECT_CFLAGS} -c $gstreplace.c ${SRC_FILES}
    if test $? -ne 0; then
        exit 1
    fi

    gcc -shared -o $gstreplace.so *.o $(pkg-config --libs gstreamer-1.0 $pkg opencv) -L${IVE_INSTALL_ROOT}/lib -lcviruntime -lcvikernel -lcvi_ive_tpu

    if test $? -ne 0; then
        exit 1
    fi

    LD_LIBRARY_PATH=${IVE_INSTALL_ROOT}/lib gst-inspect-1.0 ./$gstreplace.so

    mkdir -p ${IVE_INSTALL_ROOT}/gst-plugin
    mv $gstreplace.so ${IVE_INSTALL_ROOT}/gst-plugin
elif [ "$BUILD_TYPE" = "edb" ]
then
    CC="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-gcc"
    CXX="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-g++"
    CFLAGS="-I${TOOLCHAIN_ROOT}/aarch64-linux-gnu/libc/usr/include"
    CXXFLAGS="-I${TOOLCHAIN_ROOT}/aarch64-linux-gnu/libc/usr/include"
    LDFLAGS="-L${TOOLCHAIN_ROOT}/aarch64-linux-gnu/libc/usr/lib"
    STRIP="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-strip"
    AR="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-ar"
    RANLIB="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-gcc-ranlib"
    GLIB_CFLAGS="-I${GST_ROOT_FOLDER}/include/glib-2.0"
    GLIB_LIBS="-L${GST_ROOT_FOLDER}/lib -lgobject-2.0 -lgthread-2.0 -lgmodule-2.0 -lglib-2.0"
    GST_CFLAGS="-I${GST_ROOT_FOLDER}/include/gstreamer-1.0"
    GST_LIBS="-L${GST_ROOT_FOLDER}/lib -L${GST_ROOT_FOLDER}/lib -lgstvideo-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0"
    #AISDK_CFLAGS="-I/home/lester/Bitmain/storage/bsp_prebuilt/soc_bm1880_asic_edb_1.1.4/BM1880_AI_SDK/include"
    #AISDK_LIBS="-L/home/lester/Bitmain/storage/bsp_prebuilt/soc_bm1880_asic_edb_1.1.4/BM1880_AI_SDK/lib -lgobject-2.0 -lgthread-2.0 -lgmodule-2.0 -lglib-2.0 -lbmruntime -lbmkernel -lbmodel -lbmutils -lopencv_core -lopencv_imgcodecs"

    $CC -O3 -fPIC -DEDB $CPPFLAGS $GST_CFLAGS $GLIB_CFLAGS ${PROJECT_CFLAGS} -c $gstreplace.c ${SRC_FILES}
    if test $? -ne 0; then
      echo "Gen .o failed."
      exit 1
    fi

    $CC -shared -o $gstreplace.so *.o $GST_LIBS -L${IVE_INSTALL_ROOT}/lib -lcviruntime -lcvikernel -lcvi_ive_tpu

    if test $? -ne 0; then
       exit 1
    fi

    mkdir -p ${IVE_INSTALL_ROOT}/gst-plugin
    mv $gstreplace.so ${IVE_INSTALL_ROOT}/gst-plugin
else
    helpFunction
fi
echo "Build success."