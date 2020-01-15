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
  IVE_SRC_IMAGE_S src1, src2;
  CVI_IVE_CreateImage(handle, &src1, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  CVI_IVE_CreateImage(handle, &src2, IVE_IMAGE_TYPE_U8C1, img->width, img->height);
  memcpy(src1.pu8VirAddr[0], img->imageData, img->nChannels * img->width * img->height);
  memset(src2.pu8VirAddr[0], 255, img->nChannels * img->width * img->height);
  for (int j = img->height / 10; j < img->height * 9 / 10; j++) {
    for (int i = img->width / 10; i < img->width * 9 / 10; i++) {
      src2.pu8VirAddr[0][i + j * img->width] = 0;
    }
  }

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  printf("Run TPU Or.\n");
  CVI_IVE_Or(handle, &src1, &src2, &dst, 0);

  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dst.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_or_c.png", img, 0);
  cvReleaseImage(&img);

  // Free memory, instance
  CVI_SYS_Free(handle, &src1);
  CVI_SYS_Free(handle, &src2);
  CVI_SYS_Free(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return 0;
}