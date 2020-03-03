#include "ive.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  IVE_IMAGE_S srcL = CVI_IVE_ReadImage(handle, "tsukuba_l.png", IVE_IMAGE_TYPE_U8C1);
  IVE_IMAGE_S srcR = CVI_IVE_ReadImage(handle, "tsukuba_r.png", IVE_IMAGE_TYPE_U8C1);
  int width = srcL.u16Width;
  int height = srcL.u16Height;

  IVE_SAD_CTRL_S iveSadCtrl;
  iveSadCtrl.enMode = IVE_SAD_MODE_MB_4X4;
  iveSadCtrl.enOutCtrl = IVE_SAD_OUT_CTRL_16BIT_BOTH;
  iveSadCtrl.u16Thr = 1000;
  iveSadCtrl.u8MaxVal = 255;
  iveSadCtrl.u8MinVal = 0;

  IVE_DST_IMAGE_S dst, dst_u8, dst_thresh;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U16C1, width, height);
  CVI_IVE_CreateImage(handle, &dst_u8, IVE_IMAGE_TYPE_U8C1, width, height);
  CVI_IVE_CreateImage(handle, &dst_thresh, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run TPU SAD.\n");
  CVI_IVE_SAD(handle, &srcL, &srcR, &dst, &dst_thresh, &iveSadCtrl, 0);

  IVE_ITC_CRTL_S iveItcCtrl;
  iveItcCtrl.enType = IVE_ITC_NORMALIZE;
  CVI_IVE_ImageTypeConvert(handle, &dst, &dst_u8, &iveItcCtrl, 0);

  printf("Save to image.\n");
  CVI_IVE_WriteImage(handle, "test_sad_c.png", &dst_u8);
  CVI_IVE_WriteImage(handle, "test_sadt_c.png", &dst_thresh);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &srcL);
  CVI_SYS_FreeI(handle, &srcR);
  CVI_SYS_FreeI(handle, &dst);
  CVI_SYS_FreeI(handle, &dst_u8);
  CVI_SYS_FreeI(handle, &dst_thresh);
  CVI_IVE_DestroyHandle(handle);
  return CVI_SUCCESS;
}