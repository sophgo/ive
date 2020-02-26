# IVE Library

This is an Image Processing library using TPU on CVI1835.

**bmtap2 docker environment is recomended.**

## How to build

You'll need Middleware headers and bmtap2 library to build this project. You can let CMake get it for you from FTP server with the following command.

```
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
```

Or you can manually assign the root folder of the prebuilt libraries.

```
$ mkdir build
$ cd build
$ cmake .. -DLIBDEP_MIDDLEWARE_DIR=<middleware root folder> -DLIBDEP_BMTAP2_DIR=<bmtap2 root folder>
$ make -j8
```

SOC mode

```
$ mkdir build_soc
$ cd build
$ cmake .. -DBM_TARGET=soc -DTOOLCHAIN_ROOT_DIR=${PWD}/../../gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu  -DCMAKE_TOOLCHAIN_FILE=../toolchain/toolchain-aarch64-linux.cmake
$ make -j8
```

**Note: You'll need to connect to VPN to get prebuilt files from FTP.**

You may install the library with the following command.

```
$make install
```

## Currently supported TPU operations

1. Add
2. And
3. Filter
4. Morphology- Binary dilate/ erode
5. Or
6. Sobel X/ Y Gradient
7. Sub
   1. a - b
   2. abs(a - b)
8. Threshold
   1. Binary threshold w/o high low value
   2. Slope
9. Xor

## BMKernel known issues

1. Currently ``memcpy`` is used to put data into ``IVE_IMAGE_S``.
2. Currently TPU API does not support TL(BF16) to TG(FP32).
3. Some required API is missing,
   1. auto channel expansion
   2. div
4. Middleware include header ``cvi_type.h`` may miss some ``#define``

```
typedef unsigned short          CVI_U0Q16;

#define CVI_NOT_SUPPORTED       9
```