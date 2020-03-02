#include "ive.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// clang-format off
static char test_array[] = {
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 1, 2, 3, 4, 5, //
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7,
  6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, //
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, //
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, //
};
//clang-format on

#define TEST_W 25
#define TEST_H 20
#define CELL_SZ 5

int cpu_ref(const int res_w, const int res_h, const CVI_U32 bin_size,
            IVE_SRC_IMAGE_S *src, IVE_DST_IMAGE_S *dst_u8, IVE_DST_IMAGE_S *dst_fp32);

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  IVE_SRC_IMAGE_S src;
  CVI_IVE_CreateImage(handle, &src, IVE_IMAGE_TYPE_U8C1, TEST_W, TEST_H);
  memcpy(src.pu8VirAddr[0], test_array, TEST_W * TEST_H);
  CVI_IVE_BufFlush(handle, &src);

  int res_w = TEST_W / CELL_SZ;
  int res_h = TEST_H / CELL_SZ;
  IVE_DST_IMAGE_S dst, dst_bf16, dst_fp32;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, res_w, res_h);
  CVI_IVE_CreateImage(handle, &dst_bf16, IVE_IMAGE_TYPE_BF16C1, res_w, res_h);
  CVI_IVE_CreateImage(handle, &dst_fp32, IVE_IMAGE_TYPE_FP32C1, res_w, res_h);

  printf("Run TPU Block.\n");
  IVE_BLOCK_CTRL_S iveBlkCtrl;
  iveBlkCtrl.bin_num = 2;
  iveBlkCtrl.cell_size = CELL_SZ;
  CVI_IVE_BLOCK(handle, &src, &dst, &iveBlkCtrl, 0);
  CVI_IVE_BLOCK(handle, &src, &dst_bf16, &iveBlkCtrl, 0);

  IVE_ITC_CRTL_S iveItcCtrl;
  iveItcCtrl.enType = IVE_ITC_SATURATE;
  CVI_IVE_ImageTypeConvert(handle, &dst_bf16, &dst_fp32, &iveItcCtrl, 0);

  CVI_IVE_BufRequest(handle, &src);
  CVI_IVE_BufRequest(handle, &dst);
  CVI_IVE_BufRequest(handle, &dst_fp32);
  int ret = cpu_ref(res_w, res_h, iveBlkCtrl.bin_num, &src, &dst, &dst_fp32);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_SYS_FreeI(handle, &dst_bf16);
  CVI_SYS_FreeI(handle, &dst_fp32);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int res_w, const int res_h, const CVI_U32 bin_size,
            IVE_SRC_IMAGE_S *src, IVE_DST_IMAGE_S *dst_u8, IVE_DST_IMAGE_S *dst_fp32) {
  int ret = CVI_SUCCESS;
  float *cpu_result = malloc(res_h * res_w * sizeof(float));
  memset(cpu_result, 0, res_h * res_w * sizeof(float));
  for (size_t i = 0; i < TEST_H; i++) {
    for (size_t j = 0; j < TEST_W; j++) {
      char val = src->pu8VirAddr[0][i * TEST_W + j];
      cpu_result[(int)(i / CELL_SZ) * res_w + (int)(j / CELL_SZ)] += val;
    }
  }
  for (size_t i = 0; i < res_h; i++) {
    for (size_t j = 0; j < res_w; j++) {
      cpu_result[i * res_w + j] /= CELL_SZ * CELL_SZ * bin_size;
    }
  }
  printf("U8 check:\n");
  for (size_t i = 0; i < res_h; i++) {
    for (size_t j = 0; j < res_w; j++) {
      float f_res = cpu_result[i * res_w + j];
      int int_result = round(f_res);
      if (int_result != dst_u8->pu8VirAddr[0][i * res_w + j]) {
        printf("%d %d \n", int_result, dst_u8->pu8VirAddr[0][i * res_w + j]);
        ret = CVI_FAILURE;
        break;
      }
    }
  }
  printf("BF16 check:\n");
  for (size_t i = 0; i < res_h; i++) {
    for (size_t j = 0; j < res_w; j++) {
      if (cpu_result[i * res_w + j] != ((float*)dst_fp32->pu8VirAddr[0])[i * res_w + j]) {
        printf("%f %f \n", cpu_result[i * res_w + j],
                           ((float*)dst_fp32->pu8VirAddr[0])[i * res_w + j]);
        ret = CVI_FAILURE;
        break;
      }
    }
  }
  free(cpu_result);
  return ret;
}