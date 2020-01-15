#include "ive.h"

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

  IVE_DST_IMAGE_S dstH, dstV;
  CVI_IVE_CreateImage(handle, &dstV, IVE_IMAGE_TYPE_BF16C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstH, IVE_IMAGE_TYPE_BF16C1, img->width, img->height);

  IVE_DST_IMAGE_S dstH_u8, dstV_u8;
  CVI_IVE_CreateImage(handle, &dstH_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &dstV_u8, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  printf("Run TPU Sobel Grad.\n");
  IVE_SOBEL_CTRL_S iveSblCtrl;
  iveSblCtrl.enOutCtrl = IVE_SOBEL_OUT_CTRL_BOTH;
  CVI_IVE_Sobel(handle, &src, &dstH, &dstV, &iveSblCtrl, 0);

  IVE_ITC_CRTL_S iveItcCtrl;
  iveItcCtrl.enType = IVE_ITC_NORMALIZE;
  CVI_IVE_ImageTypeConvert(handle, &dstV, &dstV_u8, &iveItcCtrl, 0);
  CVI_IVE_ImageTypeConvert(handle, &dstH, &dstH_u8, &iveItcCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dstV_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_sobelV_c.png", img, 0);
  memcpy(img->imageData, dstH_u8.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_sobelH_c.png", img, 0);
  cvReleaseImage(&img);

  // Free memory, instance
  CVI_SYS_Free(handle, &src);
  CVI_SYS_Free(handle, &dstH);
  CVI_SYS_Free(handle, &dstH_u8);
  CVI_SYS_Free(handle, &dstV);
  CVI_SYS_Free(handle, &dstV_u8);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}