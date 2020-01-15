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

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  printf("Run TPU Filter.\n");
  CVI_S8 arr[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  IVE_FILTER_CTRL_S iveFltCtrl;
  memcpy(iveFltCtrl.as8Mask, arr, 9 * sizeof(CVI_S8));
  iveFltCtrl.u8Norm = 16;
  CVI_IVE_Filter(handle, &src, &dst, &iveFltCtrl, 0);

  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dst.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_filter_c.png", img, 0);
  cvReleaseImage(&img);

  // Free memory, instance
  CVI_SYS_Free(handle, &src);
  CVI_SYS_Free(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}
