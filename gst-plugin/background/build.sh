#!/bin/sh

gstreplace=gstcviivebackground
pkg=gstreamer-video-1.0

BUILD_TYPE=$1
BASEDIR=$(dirname $0)
CMAKE_SOURCE_DIR=${BASEDIR}
IVE_INSTALL_ROOT=${CMAKE_SOURCE_DIR}/../../install

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
    gcc -Wall -fPIC $CPPFLAGS $(pkg-config --cflags gstreamer-1.0 $pkg opencv) ${PROJECT_CFLAGS} -c $gstreplace.c ${SRC_FILES}
    if test $? -ne 0; then
        exit 1
    fi

    gcc -shared -o $gstreplace.so *.o $(pkg-config --libs gstreamer-1.0 $pkg opencv) -L${IVE_INSTALL_ROOT}/lib -lbmruntime -lbmkernel -lbmodel -lbmutils -lbmnet -lcvi_ive_tpu

    if test $? -ne 0; then
        exit 1
    fi

    LD_LIBRARY_PATH=${IVE_INSTALL_ROOT}/lib gst-inspect-1.0 ./$gstreplace.so

    mkdir -p ${IVE_INSTALL_ROOT}/gst-plugin
    mv $gstreplace.so ${IVE_INSTALL_ROOT}/gst-plugin
elif [ "$BUILD_TYPE" = "edb" ]
then
    echo "Not supported yet"
    exit 1
    # CC="/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc"
    # CXX="/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++"
    # CFLAGS="-I/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include"
    # CXXFLAGS="-I/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/include"
    # LDFLAGS="-L/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/usr/lib"
    # STRIP="/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-strip"
    # AR="/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-ar"
    # RANLIB="/home/lester/Bitmain/build/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc-ranlib"
    # GLIB_CFLAGS="-I/home/lester/Bitmain/src/gstreamer_all/dep_libs/install/usr/local/include/glib-2.0"
    # GLIB_LIBS="-L/home/lester/Bitmain/src/gstreamer_all/dep_libs/install/usr/local/lib -lgobject-2.0 -lgthread-2.0 -lgmodule-2.0 -lglib-2.0"
    # GST_CFLAGS="-I/home/lester/Bitmain/src/gstreamer_all/dep_libs/install/usr/local/include/gstreamer-1.0"
    # GST_LIBS="-L/home/lester/Bitmain/src/gstreamer_all/dep_libs/install/usr/local/lib -L/home/lester/Bitmain/src/gstreamer_all/dep_libs/install/usr/local/lib -lgstvideo-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0"
    # AISDK_CFLAGS="-I/home/lester/Bitmain/storage/bsp_prebuilt/soc_bm1880_asic_edb_1.1.4/BM1880_AI_SDK/include"
    # AISDK_LIBS="-L/home/lester/Bitmain/storage/bsp_prebuilt/soc_bm1880_asic_edb_1.1.4/BM1880_AI_SDK/lib -lgobject-2.0 -lgthread-2.0 -lgmodule-2.0 -lglib-2.0 -lbmruntime -lbmkernel -lbmodel -lbmutils -lopencv_core -lopencv_imgcodecs"

    # $CC -fPIC -DEDB $CPPFLAGS $GST_CFLAGS $GLIB_CFLAGS $AISDK_CFLAGS ${PROJECT_CFLAGS} -c $gstreplace.c ${SRC_FILES}
    # if test $? -ne 0; then
    #     exit 1
    # fi

    # $CC -shared -o $gstreplace.so *.o $GST_LIBS $AISDK_LIBS

    # if test $? -ne 0; then
    #     exit 1
    # fi

    # mv $gstreplace.so ../install/edb
else
    helpFunction
fi
