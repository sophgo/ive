#include "ive.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information
  IVE_IMAGE_S src = CVI_IVE_ReadImage(handle, "cat.png", IVE_IMAGE_TYPE_U8C1);
  int width = src.u16Width;
  int height = src.u16Height;

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU Filter.\n");
  CVI_S8 arr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  IVE_FILTER_CTRL_S iveFltCtrl;
  memcpy(iveFltCtrl.as8Mask, arr, 9 * sizeof(CVI_S8));
  iveFltCtrl.u8Norm = 16;
  CVI_IVE_Filter(handle, &src, &dst, &iveFltCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage("test_filter_c.png", &dst);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}
