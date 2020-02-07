#include "ive.h"

#include <stdio.h>
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
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, //
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
  1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, //
};
//clang-format on

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  IVE_SRC_IMAGE_S src;
  CVI_IVE_CreateImage(handle, &src, IVE_IMAGE_TYPE_U8C1, 25, 25);
  memcpy(src.pu8VirAddr[0], test_array, 625);

  IVE_DST_IMAGE_S dst, dst_bf16, dst_fp32;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, 5, 5);
  CVI_IVE_CreateImage(handle, &dst_bf16, IVE_IMAGE_TYPE_BF16C1, 5, 5);
  CVI_IVE_CreateImage(handle, &dst_fp32, IVE_IMAGE_TYPE_FP32C1, 5, 5);

  printf("Run TPU And.\n");
  IVE_BLOCK_CTRL_S iveBlkCtrl;
  iveBlkCtrl.cell_size = 5;
  CVI_IVE_BLOCK(handle, &src, &dst, &iveBlkCtrl, 0);

  printf("Result:\n");
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++) {
      printf("%d ", dst.pu8VirAddr[0][i * 5 + j]);
    }
    printf("\n");
  }

  CVI_IVE_BLOCK(handle, &src, &dst_bf16, &iveBlkCtrl, 0);

  IVE_ITC_CRTL_S iveItcCtrl;
  iveItcCtrl.enType = IVE_ITC_SATURATE;
  CVI_IVE_ImageTypeConvert(handle, &dst_bf16, &dst_fp32, &iveItcCtrl, 0);

  printf("BF16 Result:\n");
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++) {
      printf("%f ", ((float*)dst_fp32.pu8VirAddr[0])[i * 5 + j]);
    }
    printf("\n");
  }

  // Free memory, instance
  CVI_SYS_Free(handle, &src);
  CVI_SYS_Free(handle, &dst);
  CVI_SYS_Free(handle, &dst_bf16);
  CVI_SYS_Free(handle, &dst_fp32);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}