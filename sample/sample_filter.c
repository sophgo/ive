#include "ive.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Incorrect loop value. Usage: %s <file name>\n", argv[0]);
    return CVI_FAILURE;
  }
  const char *filename = argv[1];
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information. CVI_IVE_ReadImage will do the flush for you.
  IVE_IMAGE_S src = CVI_IVE_ReadImage(handle, filename, IVE_IMAGE_TYPE_U8C1);
  int width = src.u16Width;
  int height = src.u16Height;

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, width, height);

  // Setup control parameter
  CVI_S8 arr[] = {1,  4,  7, 4, 1,  4,  16, 26, 16, 4, 7, 26, 26,
                  41, 26, 7, 4, 16, 26, 16, 4,  1,  4, 7, 4,  1};
  IVE_FILTER_CTRL_S iveFltCtrl;
  iveFltCtrl.u8MaskSize = 5;  // Set the length of the mask, can be 3 or 5.
  memcpy(iveFltCtrl.as8Mask, arr, 25 * sizeof(CVI_S8));
  iveFltCtrl.u32Norm = 273;
  // Since the mask only accepts S8 values, you can turn a float mask into (1/X) * (S8 mask).
  // Then set u32Norm to X.

  printf("Run TPU Filter.\n");
  int ret = CVI_IVE_Filter(handle, &src, &dst, &iveFltCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage(handle, "sample_filter.png", &dst);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}