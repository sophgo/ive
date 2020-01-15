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
  IVE_SRC_IMAGE_S src1;
  CVI_IVE_CreateImage(handle, &src1, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  memcpy(src1.pu8VirAddr[0], img->imageData, img->nChannels * img->width * img->height);

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  printf("Run TPU Threshold.\n");
  IVE_THRESH_CTRL_S iveThreshCtrl;  // Currently a dummy variable
  iveThreshCtrl.enMode = IVE_THRESH_MODE_BINARY;
  iveThreshCtrl.u8LowThr = 170;
  iveThreshCtrl.u8MinVal = 0;
  iveThreshCtrl.u8MaxVal = 255;
  CVI_IVE_Thresh(handle, &src1, &dst, &iveThreshCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dst.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_threshold_c.png", img, 0);
  cvReleaseImage(&img);

  // Free memory, instance
  CVI_SYS_Free(handle, &src1);
  CVI_SYS_Free(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}
