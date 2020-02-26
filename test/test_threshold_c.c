#include "ive.h"

#include <stdio.h>

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

  printf("Run TPU Threshold.\n");
  IVE_THRESH_CTRL_S iveThreshCtrl;  // Currently a dummy variable
  iveThreshCtrl.enMode = IVE_THRESH_MODE_BINARY;
  iveThreshCtrl.u8LowThr = 170;
  iveThreshCtrl.u8MinVal = 0;
  iveThreshCtrl.u8MaxVal = 255;
  CVI_IVE_Thresh(handle, &src, &dst, &iveThreshCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage("test_threshold_c.png", &dst);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}
