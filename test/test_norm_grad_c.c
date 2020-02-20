#include "bmkernel/bm_kernel.h"
#include "ive.h"

#include <math.h>
#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

int main(int argc, char **argv) {
  CVI_SYS_LOGGING(argv[0]);
  // Create instance
  IVE_HANDLE handle = CVI_IVE_CreateHandle();
  printf("BM Kernel init.\n");

  // Fetch image information
  IplImage *img = cvLoadImage("cat.png", 0);
  IVE_SRC_IMAGE_S src;
  CVI_IVE_CreateImage(handle, &src, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  memcpy(src.pu8VirAddr[0], img->imageData, img->nChannels * img->width * img->height);

  IVE_DST_IMAGE_S dstH_u8, dstV_u8, dstHV_u8;
  CVI_IVE_CreateImage(handle, &dstH_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstV_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstHV_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  printf("Run norm gradient.\n");
  IVE_NORM_GRAD_CTRL_S pstNormGradCtrl;
  pstNormGradCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER;
  CVI_IVE_NormGrad(handle, &src, &dstH_u8, &dstV_u8, &dstHV_u8, &pstNormGradCtrl, 0);
  pstNormGradCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_COMBINE;
  CVI_IVE_NormGrad(handle, &src, &dstH_u8, &dstV_u8, &dstHV_u8, &pstNormGradCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dstV_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_sobelV_c.png", img, 0);
  memcpy(img->imageData, dstH_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_sobelH_c.png", img, 0);
  memcpy(img->imageData, dstHV_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_sobelHV_c.png", img, 0);
  cvReleaseImage(&img);

  // Free memory, instance
  CVI_SYS_FreeI(handle, &src);
  CVI_SYS_FreeI(handle, &dstH_u8);
  CVI_SYS_FreeI(handle, &dstV_u8);
  CVI_SYS_FreeI(handle, &dstHV_u8);
  CVI_IVE_DestroyHandle(handle);

  return CVI_SUCCESS;
}