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

  IVE_DST_IMAGE_S dst2;
  CVI_IVE_CreateImage(handle, &dst2, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU Threshold.\n");
  IVE_THRESH_CTRL_S iveThreshCtrl;  // Currently a dummy variable
  iveThreshCtrl.enMode = IVE_THRESH_MODE_BINARY;
  iveThreshCtrl.u8LowThr = 170;
  iveThreshCtrl.u8MinVal = 0;
  iveThreshCtrl.u8MaxVal = 255;
  CVI_IVE_Thresh(handle, &src, &dst, &iveThreshCtrl, 0);

  // write result to disk
  CVI_IVE_WriteImage("test_morph_thresh_c.png", &dst);

  printf("Run TPU Dilate.\n");
  CVI_U8 arr[] = {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
  IVE_DILATE_CTRL_S iveDltCtrl;
  memcpy(iveDltCtrl.au8Mask, arr, 25 * sizeof(CVI_U8));
  CVI_IVE_Dilate(handle, &dst, &dst2, &iveDltCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage("test_dilate_c.png", &dst2);

  printf("Run TPU Erode.\n");
  IVE_ERODE_CTRL_S iveErdCtrl = iveDltCtrl;
  CVI_IVE_Erode(handle, &dst, &dst2, &iveErdCtrl, 0);
  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage("test_erode_c.png", &dst2);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dst);
  CVI_SYS_FreeI(handle, &dst2);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}
