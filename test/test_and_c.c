#include "ive.h"

#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src1, IVE_SRC_IMAGE_S *src2, IVE_DST_IMAGE_S *dst);

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
  memset(src2.pu8VirAddr[0], 0, img->nChannels * img->width * img->height);
  for (int j = img->height / 10; j < img->height * 9 / 10; j++) {
    for (int i = img->width / 10; i < img->width * 9 / 10; i++) {
      src2.pu8VirAddr[0][i + j * img->width] = 255;
    }
  }

  IVE_DST_IMAGE_S dst;
  CVI_IVE_CreateImage(handle, &dst, IVE_IMAGE_TYPE_U8C1, img->width, img->height);

  printf("Run TPU And.\n");
  CVI_IVE_And(handle, &src1, &src2, &dst, 0);

  int ret = cpu_ref(img->nChannels, &src1, &src2, &dst);

  // write result to disk
  printf("Save to image.\n");
  memcpy(img->imageData, dst.pu8VirAddr[0], img->nChannels * img->width * img->height);
  cvSaveImage("test_and_c.png", img, 0);
  cvReleaseImage(&img);

  // Free memory, instance
  CVI_SYS_Free(handle, &src1);
  CVI_SYS_Free(handle, &src2);
  CVI_SYS_Free(handle, &dst);
  CVI_IVE_DestroyHandle(handle);

  return ret;
}

int cpu_ref(const int channels, IVE_SRC_IMAGE_S *src1, IVE_SRC_IMAGE_S *src2,
            IVE_DST_IMAGE_S *dst) {
  int ret = CVI_SUCCESS;
  CVI_U8 *src1_ptr = src1->pu8VirAddr[0];
  CVI_U8 *src2_ptr = src2->pu8VirAddr[0];
  CVI_U8 *dst_ptr = dst->pu8VirAddr[0];
  for (size_t i = 0; i < channels * src1->u16Width * src1->u16Height; i++) {
    int res = src1_ptr[i] & src2_ptr[i];
    if (res != dst_ptr[i]) {
      printf("[%lu] %d & %d = TPU %d, CPU %d\n", i, src1_ptr[i], src2_ptr[i], dst_ptr[i],
             (int)src1_ptr[i] + src2_ptr[i]);
      ret = CVI_FAILURE;
      break;
    }
  }
  return ret;
}