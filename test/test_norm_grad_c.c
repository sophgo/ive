#include "bmkernel/bm_kernel.h"
#include "ive.h"

#include <math.h>
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

  IVE_DST_IMAGE_S dstH_u8, dstV_u8, dstHV_u8;
  CVI_IVE_CreateImage(handle, &dstH_u8, IVE_IMAGE_TYPE_U8C1, width, height);
  CVI_IVE_CreateImage(handle, &dstV_u8, IVE_IMAGE_TYPE_U8C1, width, height);
  CVI_IVE_CreateImage(handle, &dstHV_u8, IVE_IMAGE_TYPE_U8C1, width, height);

  printf("Run norm gradient.\n");
  IVE_NORM_GRAD_CTRL_S pstNormGradCtrl;
  pstNormGradCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER;
  CVI_IVE_NormGrad(handle, &src, &dstH_u8, &dstV_u8, &dstHV_u8, &pstNormGradCtrl, 0);
  pstNormGradCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_COMBINE;
  CVI_IVE_NormGrad(handle, &src, &dstH_u8, &dstV_u8, &dstHV_u8, &pstNormGradCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  CVI_IVE_WriteImage("test_normV_c.png", &dstV_u8);
  CVI_IVE_WriteImage("test_normH_c.png", &dstH_u8);
  CVI_IVE_WriteImage("test_normHV_c.png", &dstHV_u8);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dstH_u8);
  CVI_SYS_FreeI(handle, &dstV_u8);
  CVI_SYS_FreeI(handle, &dstHV_u8);
  CVI_IVE_DestroyHandle(handle);

  return CVI_SUCCESS;
}