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

  printf("Run TPU Direct Copy.\n");
  IVE_DMA_CTRL_S iveDmaCtrl;
  iveDmaCtrl.enMode = IVE_DMA_MODE_DIRECT_COPY;
  CVI_IVE_DMA(handle, &src, &dst, &iveDmaCtrl, 0);

  IVE_DST_IMAGE_S dst2;
  CVI_IVE_CreateImage(handle, &dst2, IVE_IMAGE_TYPE_U8C1, img->width * 4, img->height * 4);

  printf("Run TPU Interval Copy.\n");
  iveDmaCtrl.enMode = IVE_DMA_MODE_INTERVAL_COPY;
  iveDmaCtrl.u8HorSegSize = 4;
  iveDmaCtrl.u8VerSegRows = 4;
  CVI_IVE_DMA(handle, &src, &dst2, &iveDmaCtrl, 0);
  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dst.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_dcopy_c.png", img, 0);
  cvReleaseImage(&img);

  IplImage *img_interval;
  CvSize img_size = cvSize(dst2.u16Width, dst2.u16Height);
  img_interval = cvCreateImage(img_size, IPL_DEPTH_8U, 1);
  memcpy(img_interval->imageData, dst2.pu8VirAddr[0], img_interval->nChannels * img_interval->width * img_interval->height);
  cvSaveImage("test_icopy_c.png", img_interval, 0);
  cvReleaseImage(&img_interval);

  // Free memory, instance
  CVI_SYS_Free(handle, &src);
  CVI_SYS_Free(handle, &dst);
  CVI_SYS_Free(handle, &dst2);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}