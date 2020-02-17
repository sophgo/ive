#include "ive.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int cpu_ref(const int res_w, const int res_h, const int cell_size,
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
  int cell_size = 4;
  switch (iveSadCtrl.enMode) {
    case IVE_SAD_MODE_MB_4X4:
    cell_size = 4;
    break;
    case IVE_SAD_MODE_MB_8X8:
    cell_size = 8;
    break;
    case IVE_SAD_MODE_MB_16X16:
    cell_size = 16;
    break;
    default:
      printf("Not supported SAD mode.\n");
    break;
  }
  int res_w = TEST_W / cell_size;
  int res_h = TEST_H / cell_size;
  IVE_DST_IMAGE_S dst, dst_thresh;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U16C1, res_w, res_h);
  CVI_IVE_CreateImage(handle, &dst_thresh, IVE_IMAGE_TYPE_U8C1, res_w, res_h);

  printf("Run TPU SAD.\n");
  CVI_IVE_SAD(handle, &src, &src2, &dst, &dst_thresh, &iveSadCtrl, 0);

  int ret = cpu_ref(res_w, res_h, cell_size, iveSadCtrl.u16Thr, iveSadCtrl.u8MinVal,
                    iveSadCtrl.u8MaxVal, &src, &src2, &dst, &dst_thresh);

  // Free memory, instance
  CVI_SYS_Free(handle, &src);
  CVI_SYS_Free(handle, &src2);
  CVI_SYS_Free(handle, &dst);
  CVI_SYS_Free(handle, &dst_thresh);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int res_w, const int res_h, const int cell_size,
            const int threshold, const int min, const int max,
            IVE_SRC_IMAGE_S *src, IVE_SRC_IMAGE_S *src2,
            IVE_DST_IMAGE_S *dst, IVE_DST_IMAGE_S *dst_thresh) {
  int ret = CVI_SUCCESS;
  float *cpu_result = malloc(res_h * res_w * sizeof(float));
  memset(cpu_result, 0, res_h * res_w * sizeof(float));
  for (size_t i = 0; i < TEST_H; i++) {
    for (size_t j = 0; j < TEST_W; j++) {
      int val = src->pu8VirAddr[0][i * TEST_W + j];
      int val2 = src2->pu8VirAddr[0][i * TEST_W + j];
      cpu_result[(int)(i / cell_size) * res_w + (int)(j / cell_size)] += abs(val - val2);
    }
  }

  if (dst->enType == IVE_IMAGE_TYPE_U16C1) {
    printf("SAD result is U16\n");
    unsigned short *dst_addr = (unsigned short*)dst->pu8VirAddr[0];
    for (size_t i = 0; i < res_h; i++) {
      for (size_t j = 0; j < res_w; j++) {
        float f_res = cpu_result[i * res_w + j];
        int int_result = round(f_res);

        if (int_result != dst_addr[i * res_w + j]) {
          printf("[%lu, %lu] %d %d \n", j, i, int_result, dst_addr[i * res_w + j]);
          ret = CVI_FAILURE;
          break;
        }
      }
    }
  } else {
    printf("SAD result is U8\n");
    unsigned char *dst_addr = (unsigned char*)dst->pu8VirAddr[0];
    for (size_t i = 0; i < res_h; i++) {
      for (size_t j = 0; j < res_w; j++) {
        float f_res = cpu_result[i * res_w + j];
        int int_result = f_res;

        if (int_result != dst_addr[i * res_w + j]) {
          printf("[%lu, %lu] %d %d \n", j, i, int_result, dst_addr[i * res_w + j]);
          ret = CVI_FAILURE;
          break;
        }
      }
    }
  }

  unsigned char *dst_thresh_addr = dst_thresh->pu8VirAddr[0];
  for (size_t i = 0; i < res_h; i++) {
    for (size_t j = 0; j < res_w; j++) {
      int value = cpu_result[i * res_w + j] >= threshold ? max : min;
      if (value != dst_thresh_addr[i * res_w + j]) {
        printf("[%lu, %lu] %f %d \n", j, i, cpu_result[i * res_w + j],
                                            (int)dst_thresh_addr[i * res_w + j]);
        ret = CVI_FAILURE;
        break;
      }
    }
  }
  free(cpu_result);
#ifdef CVI_PRINT_RESULT
  printf("Original:\n");
  unsigned short *dstu16 = (unsigned short*)dst.pu8VirAddr[0];
  for (size_t i = 0; i < res_h; i++) {
    for (size_t j = 0; j < res_w; j++) {
      printf("%d ", (int)dstu16[j + i * res_w]);
    }
    printf("\n");
  }
  printf("Threshold:\n");
  for (size_t i = 0; i < res_h; i++) {
    for (size_t j = 0; j < res_w; j++) {
      printf("%d ", dst_thresh.pu8VirAddr[0][j + i * res_w]);
    }
    printf("\n");
  }
#endif
  return ret;
}