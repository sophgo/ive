# IVE Library

This is an Image Processing library using TPU on CVI1835.

## How to build

**Cmodel library, you can get it from bmtap2. The compiled library should be placed under ``prebuilt/cmodel_bm1880v2``. bmtap2 docker environment is recomended.**

**Middleware headers are required. Put them under ``prebuilt/middleware``.**

```
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
```

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
8. Threshold
   1. Binary threshold w/o high low value
   2. Slope
9. Xor

## BMKernel known issues

1. Currently ``memcpy`` is used to put data into ``IVE_IMAGE_S``.
2. Currently TPU API does not support TL(BF16) to TG(FP32).
3. Some required API is missing,
   1. auto channel expansion
   2. atan2
   3. sqrt
   4. div