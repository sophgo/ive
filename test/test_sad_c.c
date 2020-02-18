#include "ive.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

// clang-format off
static char test_array[] = {
  1, 2, 3, 4, 2, 2, 2, 2, 7, 7, 7, 7, 7, 6, 8, 3, 9, 9, 9, 9, 1, 2, 3, 4,
  1, 2, 3, 4, 2, 2, 2, 2, 7, 7, 7, 7, 7, 6, 8, 3, 9, 9, 9, 9, 1, 2, 3, 4,
  1, 2, 3, 4, 2, 2, 2, 2, 7, 7, 7, 7, 7, 6, 8, 3, 9, 9, 9, 9, 1, 2, 3, 4,
  1, 2, 3, 4, 2, 2, 2, 2, 7, 7, 7, 7, 7, 6, 8, 3, 9, 9, 9, 9, 1, 2, 3, 4,
  1, 2, 3, 4, 2, 2, 2, 2, 7, 7, 7, 7, 7, 6, 8, 3, 9, 9, 9, 9, 1, 2, 3, 4, //
  6, 6, 6, 6, 4, 4, 4, 4, 1, 2, 3, 4, 7, 6, 8, 3, 1, 2, 3, 4, 7, 7, 7, 7,
  6, 6, 6, 6, 4, 4, 4, 4, 1, 2, 3, 4, 7, 6, 8, 3, 1, 2, 3, 4, 7, 7, 7, 7,
  6, 6, 6, 6, 4, 4, 4, 4, 1, 2, 3, 4, 7, 6, 8, 3, 1, 2, 3, 4, 7, 7, 7, 7,
  6, 6, 6, 6, 4, 4, 4, 4, 1, 2, 3, 4, 7, 6, 8, 3, 1, 2, 3, 4, 7, 7, 7, 7,
  6, 6, 6, 6, 4, 4, 4, 4, 1, 2, 3, 4, 7, 6, 8, 3, 1, 2, 3, 4, 7, 7, 7, 7, //
  1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4, //
  1, 2, 3, 4, 1, 2, 3, 4, 8, 8, 8, 8, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 8, 8, 8, 8, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 8, 8, 8, 8, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 8, 8, 8, 8, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 8, 8, 8, 8, 7, 6, 8, 3, 1, 2, 3, 4, 1, 2, 3, 4, //
};
static char test_array2[] = {
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, //
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, //
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, //
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, //
};
//clang-format on

#define TEST_W 24
#define TEST_H 20

int cpu_ref(const int width, const int height, const int window_size,
            const int threshold, const int min, const int max,
            IVE_SRC_IMAGE_S *src, IVE_SRC_IMAGE_S *src2,
            IVE_DST_IMAGE_S *dst, IVE_DST_IMAGE_S *dst_thresh);

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  IVE_SRC_IMAGE_S src, src2;
  CVI_IVE_CreateImage(handle, &src, IVE_IMAGE_TYPE_U8C1, TEST_W, TEST_H);
  memcpy(src.pu8VirAddr[0], test_array, TEST_W * TEST_H);
  CVI_IVE_CreateImage(handle, &src2, IVE_IMAGE_TYPE_U8C1, TEST_W, TEST_H);
  memcpy(src2.pu8VirAddr[0], test_array2, TEST_W * TEST_H);

  IVE_SAD_CTRL_S iveSadCtrl;
  iveSadCtrl.enMode = IVE_SAD_MODE_MB_4X4;
  iveSadCtrl.enOutCtrl = IVE_SAD_OUT_CTRL_16BIT_BOTH;
  iveSadCtrl.u16Thr = 25;
  iveSadCtrl.u8MaxVal = 128;
  iveSadCtrl.u8MinVal = 0;
  int window_size = 4;
  switch (iveSadCtrl.enMode) {
    case IVE_SAD_MODE_MB_4X4:
    window_size = 4;
    break;
    case IVE_SAD_MODE_MB_8X8:
    window_size = 8;
    break;
    case IVE_SAD_MODE_MB_16X16:
    window_size = 16;
    break;
    default:
      printf("Not supported SAD mode.\n");
    break;
  }

  IVE_DST_IMAGE_S dst, dst_thresh;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U16C1, TEST_W, TEST_H);
  CVI_IVE_CreateImage(handle, &dst_thresh, IVE_IMAGE_TYPE_U8C1, TEST_W, TEST_H);

  printf("Run TPU SAD.\n");
  CVI_IVE_SAD(handle, &src, &src2, &dst, &dst_thresh, &iveSadCtrl, 0);

  int ret = cpu_ref(TEST_W, TEST_H, window_size, iveSadCtrl.u16Thr, iveSadCtrl.u8MinVal,
                    iveSadCtrl.u8MaxVal, &src, &src2, &dst, &dst_thresh);

  // Free memory, instance
  CVI_SYS_Free(handle, &src);
  CVI_SYS_Free(handle, &src2);
  CVI_SYS_Free(handle, &dst);
  CVI_SYS_Free(handle, &dst_thresh);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int width, const int height, const int window_size,
            const int threshold, const int min, const int max,
            IVE_SRC_IMAGE_S *src, IVE_SRC_IMAGE_S *src2,
            IVE_DST_IMAGE_S *dst, IVE_DST_IMAGE_S *dst_thresh) {
  int ret = CVI_SUCCESS;
  float *cpu_result = malloc(height * width * sizeof(float));
  memset(cpu_result, 0, height * width * sizeof(float));
  CVI_U32 pad_1 = window_size / 2;
  CVI_U32 pad_0 = pad_1 - 1;
  for (size_t i = 0; i < height - window_size + 1; i++) {
    for (size_t j = 0; j < width - window_size + 1; j++) {
      int total = 0;
      for (size_t a = 0; a < window_size; a++) {
        for (size_t b = 0; b < window_size; b++) {
          int val = src->pu8VirAddr[0][(i + a) * width + (j + b)];
          int val2 = src2->pu8VirAddr[0][(i + a) * width + (j + b)];
          total += abs(val - val2);
        }
      }
      cpu_result[(i + pad_0) * width + j + pad_0] = total;
    }
  }

  if (dst->enType == IVE_IMAGE_TYPE_U16C1) {
    printf("SAD result is U16\n");
    unsigned short *dst_addr = (unsigned short*)dst->pu8VirAddr[0];
    for (size_t i = pad_0; i < height - pad_1; i++) {
      for (size_t j = pad_0; j < width - pad_1; j++) {
        float f_res = cpu_result[i * width + j];
        int int_result = round(f_res);

        if (int_result != dst_addr[i * width + j]) {
          printf("[%lu, %lu] %d %d \n", j, i, int_result, dst_addr[i * width + j]);
          ret = CVI_FAILURE;
          break;
        }
      }
    }
  } else {
    printf("SAD result is U8\n");
    unsigned char *dst_addr = (unsigned char*)dst->pu8VirAddr[0];
    for (size_t i = pad_0; i < height - pad_1; i++) {
      for (size_t j = pad_0; j < width - pad_1; j++) {
        float f_res = cpu_result[i * width + j];
        int int_result = f_res;

        if (int_result != dst_addr[i * width + j]) {
          printf("[%lu, %lu] %d %d \n", j, i, int_result, dst_addr[i * width + j]);
          ret = CVI_FAILURE;
          break;
        }
      }
    }
  }

  unsigned char *dst_thresh_addr = dst_thresh->pu8VirAddr[0];
  for (size_t i = pad_0; i < height - pad_1; i++) {
    for (size_t j = pad_0; j < width - pad_1; j++) {
      int value = cpu_result[i * width + j] >= threshold ? max : min;
      if (value != dst_thresh_addr[i * width + j]) {
        printf("[%lu, %lu] %f %d \n", j, i, cpu_result[i * width + j],
                                            (int)dst_thresh_addr[i * width + j]);
        ret = CVI_FAILURE;
        break;
      }
    }
  }
  free(cpu_result);
#ifdef CVI_PRINT_RESULT
  printf("Original:\n");
  unsigned short *dstu16 = (unsigned short*)dst->pu8VirAddr[0];
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      printf("%d ", (int)dstu16[j + i * width]);
    }
    printf("\n");
  }
  printf("Threshold:\n");
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      printf("%d ", dst_thresh->pu8VirAddr[0][j + i * width]);
    }
    printf("\n");
  }
#endif
  return ret;
}