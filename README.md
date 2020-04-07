# IVE Library

This is an Image Processing library using TPU on CVI1835.

**bmtap2 docker environment is recomended.**

## How to build

### Requirements

1. Middleware headers
2. MLIR SDK
3. Tracer lib

Tracer lib can be get from GitLab at ``http://10.34.33.3:8480/sys_app/tracer``. Put the Tracer lib under the ``3rdparty`` folder.

```
$ mkdir build
$ cd build
$ cmake -G Ninja .. -DMLIR_SDK_ROOT=${PWD}/../../cvitek_mlir \
                    -DMIDDLEWARE_SDK_ROOT=${PWD}/../../middleware \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja -j8
```

SOC mode

```
$ mkdir build_soc
$ cd build
$ cmake -G Ninja .. -DCVI_TARGET=soc \
                    -DTOOLCHAIN_ROOT_DIR=${PWD}/../../gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu \
                    -DCMAKE_TOOLCHAIN_FILE=${PWD}/../toolchain/toolchain-aarch64-linux.cmake \
                    -DMLIR_SDK_ROOT=${PWD}/../../cvitek_tpu_sdk \
                    -DMIDDLEWARE_SDK_ROOT=${PWD}/../../middleware \
                    -DCMAKE_BUILD_TYPE=Release
$ ninja -j8
```

You may install the library with the following command.

```
$ninja install
```

## Currently supported TPU operations

1. Add
2. And
3. Block
4. Copy
5. Filter
6. HOG
7. Morphology- Binary dilate/ erode
8. Normalize gradient
9. Or
10. SAD
11. Sigmoid
12. Sobel X/ Y Gradient
13. Sub
   1. a - b
   2. abs(a - b)
14. Threshold
   3. Binary threshold w/o high low value
   4. Slope
15. Xor

## Currently supported NEON operations

1. U16 related image conversion.
2. U16, S16 to U8, S8 threshold.

## BMKernel known issues

1. Currently ``memcpy`` is used to put data into ``IVE_IMAGE_S``.
2. Currently TPU API does not support TL(BF16) to TG(FP32).
3. Some required API is missing,
   1. ~~auto channel expansion~~
   2. div
4. ~~Middleware include header ``cvi_type.h`` may miss some ``#define``~~